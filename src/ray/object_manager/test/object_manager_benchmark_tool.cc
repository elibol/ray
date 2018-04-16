#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <numeric>

#include "ray/object_manager/object_manager.h"
#include "object_manager_test_common.h"

namespace ray {
namespace object_manager {
namespace test {

std::string store_executable;

class ObjectManagerBenchmarkTool {

 public:

  ObjectManagerBenchmarkTool(std::string node_ip_address,
                             std::string redis_address,
                             int redis_port,
                             int max_sends,
                             int max_receives,
                             std::string mode,
                             uint64_t object_chunk_size,
                             const std::string &store_gigabytes_memory) {

    object_manager_service_1.reset(new boost::asio::io_service());
    work_.reset(new boost::asio::io_service::work(main_service));

    // start store
    store_id = StartStore(mode, store_executable, store_gigabytes_memory);

    // start first server
    gcs_client_1 = std::shared_ptr<gcs::AsyncGcsClient>(new gcs::AsyncGcsClient());
    ObjectManagerConfig om_config_1;
    om_config_1.store_socket_name = store_id;
    om_config_1.max_sends = max_sends;
    om_config_1.max_receives = max_receives;
    om_config_1.object_chunk_size = object_chunk_size;
    server1.reset(new test::MockServer(main_service,
                                     node_ip_address,
                                     redis_address,
                                     redis_port,
                                     std::move(object_manager_service_1),
                                     om_config_1,
                                     gcs_client_1));

    // connect to stores.
    ARROW_CHECK_OK(client1.Connect(store_id, "", PLASMA_DEFAULT_RELEASE_DELAY));
  }

  void TearDown() {
    main_service.post([this](){
      sleep(1);
      StopStore(store_id);
      main_service.stop();
      ARROW_CHECK_OK(client1.Disconnect());
      this->server1.reset();
    });
    work_.reset();
  }

  void ConnectAndExecute(std::string mode, uint64_t object_size, int num_objects, int num_trials, uint64_t skip_k) {
    skip_k_ = skip_k;
    RAY_LOG(INFO) << "creating " << num_objects*num_trials
                  << " empty objects of size " << object_size;
    if (mode != "receive"){
      // Create the objects to send before connecting.
      // The remote manager will start timing as soon as the sender connects,
      // so we want to make sure we're not timing object creation.
      for (int trial=0; trial < num_trials; ++trial){
        send_object_ids.emplace_back(std::unordered_set<ObjectID, UniqueIDHasher>());
        for (int i=0;i<num_objects;++i) {
          ObjectID oid = test::WriteDataToClient(client1, object_size, 0);
          send_object_ids[trial].insert(oid);
          ignore_send_ids.insert(oid);
        }
        RAY_LOG(INFO) << "trial " << trial << " created.";
      }
    }
    ClientID client_id_1 = gcs_client_1->client_table().GetLocalClientId();
    RAY_LOG(INFO) << "local client_id " << client_id_1;
    gcs_client_1->client_table().RegisterClientAddedCallback(
        [this, client_id_1, mode, object_size, num_objects, num_trials]
            (gcs::AsyncGcsClient *client, const ClientID &id, const ClientTableDataT &data) {
          ClientID parsed_id = ClientID::from_binary(data.client_id);
          if (!(parsed_id == client_id_1)){
            Execute(parsed_id, mode, object_size, num_objects, num_trials);
          }
    });
  }

  void Execute(ClientID remote_client_id, std::string mode, uint64_t object_size, int num_objects, int num_trials){
    RAY_LOG(INFO) << "remote client_id " << remote_client_id;
    ray::Status status = ray::Status::OK();
    if (mode == "receive"){
      ExecuteReceiveMode(remote_client_id, object_size, num_objects, num_trials);
    } else if (mode == "send"){
      ExecuteSendMode(remote_client_id, object_size, num_objects, num_trials);
    } else if (mode == "bidirectional"){
      ExecuteBidirectionalMode(remote_client_id, mode, object_size, num_objects, num_trials);
    } else {
      RAY_LOG(FATAL) << mode << " is not a supported mode.";
    }
  }

  void ExecuteSendMode(ClientID remote_client_id, uint64_t object_size, int num_objects, int num_trials){
    ray::Status status =
        server1->object_manager_.SubscribeObjAdded(
            [this, remote_client_id, object_size, num_objects, num_trials](const ObjectInfoT &object_info) {
              ObjectID object_id = ObjectID::from_binary(object_info.object_id);
              if (ignore_send_ids.count(object_id) != 0) {
                // Ignore objects being sent by this manager.
                return;
              }
              // Everything we receive is an init object in this mode.
              RAY_LOG(DEBUG) << "received " << object_id;
              BeginTrial(remote_client_id);
              int64_t elapsed = current_time_ms() - start_time;
              RAY_LOG(DEBUG) << "trial=" << trial_count
                             << " elapsed=" << elapsed
                             << " object_id=" << object_id;
              for (auto oid : send_object_ids[trial_count]) {
                RAY_LOG(DEBUG) << "sent " << oid;
              }
              trial_count += 1;
              if (trial_count >= num_trials){
                TearDown();
              }
            }
        );
    RAY_CHECK_OK(status);
  }

  void ExecuteReceiveMode(ClientID remote_client_id, uint64_t object_size, int num_objects, int num_trials){
    SendInit(remote_client_id);
    // start timer now since the sender will start sending as soon as it receives
    // the small object.
    ray::Status status =
        server1->object_manager_.SubscribeObjAdded(
            [this, remote_client_id, object_size,
                num_objects, num_trials](const ObjectInfoT &object_info) {
              ObjectID object_id = ObjectID::from_binary(object_info.object_id);
              if (init_objects.count(object_id) > 0){
                // ignore the initial object we sent out to start the experiment.
                // start the timer here since we will certainly register object added
                // before the remote object manager does.
                start_time = current_time_ms();
                return;
              }
              // record stats
              v1.push_back(object_id);
              receive_times.push_back(current_time_ms());
              if ((int)v1.size() == num_objects) {
                TrialComplete(object_size, num_objects);
                if (trial_count < num_trials) {
                  SendInit(remote_client_id);
                } else {
                  OutputStats();
                  TearDown();
                }
              }
            }
        );
    RAY_CHECK_OK(status);
  }

  void ExecuteBidirectionalMode(ClientID remote_client_id, std::string mode, uint64_t object_size, int num_objects, int num_trials){
    ray::Status status;
    status = server1->object_manager_.SubscribeObjAdded(
        [this, remote_client_id, object_size, num_objects, num_trials](const ObjectInfoT &object_info) {
          ObjectID object_id = ObjectID::from_binary(object_info.object_id);
          if (init_objects.count(object_id) > 0){
            // Ignore the init object sent to the remote manager.
            return;
          }
          if (ignore_send_ids.count(object_id) != 0) {
            // Ignore objects being sent by this manager.
            return;
          }
          if (object_info.data_size == 1){
            // This is an init object.
            RAY_LOG(INFO) << "init received " << object_id;
            remote_trial_ready = true;
            if (remote_trial_ready && local_trial_ready){
              // We're waiting on remote end, and the remote end
              // is ready, so begin sending data.
              BeginTrial(remote_client_id);
            }
            // Don't process this as a receive.
            return;
          }

          // Record stats.
          v1.push_back(object_id);
          RAY_LOG(INFO) << "processed " << object_id;
          receive_times.push_back(current_time_ms());

          if ((int)v1.size() == num_objects) {
            TrialComplete(object_size, num_objects);
            if (trial_count < num_trials) {
              local_trial_ready = true;
              SendInit(remote_client_id);
              if (remote_trial_ready && local_trial_ready){
                // If the remote end has already indicated that it's ready, then begin
                // sending data as the remote end is waiting on us.
                BeginTrial(remote_client_id);
              }
            } else {
              OutputStats();
              TearDown();
            }

          }

        }
    );
    RAY_CHECK_OK(status);
    // We won't begin executing object added handlers until we're actually "ready,"
    // so mark ourselves as ready before subscribing as we'll be ready to begin sending
    // data as soon as we receive notice that the remote end is ready.
    // This will synchronize starting time because we'll be waiting until the remote
    // end indicates it's ready, and it won't do that until it's actually executing
    // the next few lines of code.
    local_trial_ready = true;
    SendInit(remote_client_id);
  }

  void SendInit(ClientID remote_client_id){
    // send a small object to remote om to indicate that this om is
    // ready to begin the next trial.
    ObjectID oid = test::WriteDataToClient(client1, 1, 0);
    init_objects.insert(oid);
    Status push_status = server1->object_manager_.Push(oid, remote_client_id);
    RAY_LOG(INFO) << "init sent " << oid;
  }

  void BeginTrial(ClientID remote_client_id){
    RAY_LOG(INFO) << "begin trial " << trial_count << " " << current_time_ms();
    local_trial_ready = false;
    remote_trial_ready = false;
    start_time = current_time_ms();
    for (auto oid : send_object_ids[trial_count]) {
      ray::Status async_status = server1->object_manager_.Push(oid, remote_client_id);
      RAY_CHECK_OK(async_status);
    }
  }

  void TrialComplete(uint64_t object_size, int num_objects){
    double_t elapsed = current_time_ms() - start_time;
    for (uint i=0;i<v1.size();++i) {
      RAY_LOG(DEBUG) << "received " << v1[i] << " " << receive_times[i];
    }
    double_t gbits = (double)object_size*num_objects*8.0/1000.0/1000.0/1000.0;
    double_t gbits_sec = gbits/(elapsed/1000.0);
    int64_t min_time = *std::min_element(receive_times.begin(), receive_times.end());
    int64_t max_time = *std::max_element(receive_times.begin(), receive_times.end());
    if (elapsed != 0){
      // Only record trial if it took longer than 0 ms.
      elapsed_stats_.push_back(elapsed);
      gbits_sec_stats_.push_back(gbits_sec);
      duration_stats_.push_back((double)max_time-(double)min_time);
      RAY_LOG(DEBUG) << "elapsed milliseconds " << elapsed;
      RAY_LOG(DEBUG) << "GBits transferred " << gbits;
      RAY_LOG(DEBUG) << "GBits/sec " << gbits_sec;
      RAY_LOG(DEBUG) << "max=" << max_time << " min=" << min_time;
      RAY_LOG(DEBUG) << "max-min time " << (max_time-min_time);
      RAY_LOG(INFO) << "TrialComplete " << trial_count;
    }
    trial_count += 1;
    // clear stats
    v1.clear();
    receive_times.clear();
  }

  void OutputStats(){
    std::pair<double_t,double_t> elapsed_stat = mean_std(elapsed_stats_, skip_k_);
    std::pair<double_t,double_t> gbits_sec_stat = mean_std(gbits_sec_stats_, skip_k_);
    std::pair<double_t,double_t> duration_stat = mean_std(duration_stats_, skip_k_);
    RAY_LOG(INFO) << "elapsed milliseconds "
                  << "mean=" << elapsed_stat.first
                  << " std=" << elapsed_stat.second;
    RAY_LOG(INFO) << "GBits/sec "
                  << "mean=" << gbits_sec_stat.first
                  << " std=" << gbits_sec_stat.second;
    RAY_LOG(INFO) << "max-min time "
                  << "mean=" << duration_stat.first
                  << " std=" << duration_stat.second;
  }

  boost::asio::io_service main_service;

 protected:
  std::string store_id;
  std::vector<std::unordered_set<ObjectID, UniqueIDHasher>> send_object_ids;
  std::unordered_set<ObjectID, UniqueIDHasher> ignore_send_ids;

  std::thread p;
  std::unique_ptr<boost::asio::io_service> object_manager_service_1;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_1;
  std::unique_ptr<test::MockServer> server1;

  plasma::PlasmaClient client1;

  std::vector<ObjectID> v1;
  std::vector<int64_t> receive_times;
  bool local_trial_ready = false;
  bool remote_trial_ready = false;

  // experiment-specific variables
  int trial_count = 0;
  int64_t start_time;
  std::unordered_set<ObjectID, UniqueIDHasher> init_objects;
  uint64_t skip_k_ = 0;

  // stats
  std::vector<double_t> elapsed_stats_;
  std::vector<double_t> gbits_sec_stats_;
  std::vector<double_t> duration_stats_;

  std::unique_ptr<boost::asio::io_service::work> work_;
};


} // namespace test
} // namespace object_manager
}  // namespace ray

int main(int argc, char **argv) {
  const std::string node_ip_address = std::string(argv[1]);
  const std::string redis_address = std::string(argv[2]);
  int redis_port = std::stoi(argv[3]);
  ray::object_manager::test::store_executable = std::string(argv[4]);
  const std::string mode = std::string(argv[5]);

  uint64_t object_size = (uint64_t) std::stol(argv[6]);
  int num_objects = std::stoi(argv[7]);
  int num_trials = std::stoi(argv[8]);

  int max_sends = std::stoi(argv[9]);
  int max_receives = std::stoi(argv[10]);
  uint64_t object_chunk_size = (uint64_t) std::stol(argv[11]);
  const std::string store_gigabytes_memory = std::string(argv[12]);
  uint64_t skip_k = (uint64_t) std::stol(argv[13]);

  // Compute num chunks (see object_buffer_pool.cc for equivalent computation).
  uint64_t num_chunks = static_cast<uint64_t>(ceil(static_cast<double>(object_size) / object_chunk_size));
  uint64_t require_memory = static_cast<uint64_t>(object_size*num_objects*num_trials);
  uint64_t store_bytes_memory = static_cast<uint64_t>(std::stoi(store_gigabytes_memory)*std::pow(10, 9));

  RAY_LOG(INFO) <<"\n"
      << "node_ip_address= " << node_ip_address << "\n"
      << "redis_address=   " << redis_address << "\n"
      << "redis_port=      " << redis_port << "\n"
      << "store_executable=" << ray::object_manager::test::store_executable << "\n"
      << "store_bytes_memory=      " << store_bytes_memory << "\n"
      << "require_memory(computed)=" << require_memory << "\n"

      << "\n"
      << "mode=                " << mode << "\n"
      << "num_trials=          " << num_trials << "\n"
      << "skip_k=              " << skip_k << "\n"
      << "num_objects=         " << num_objects << "\n"
      << "object_size=         " << object_size << "\n"
      << "object_chunk_size=   " << object_chunk_size << "\n"
      << "num_chunks(computed)=" << num_chunks << "\n"

      << "\n"
      << "   max_sends=" << max_sends << "\n"
      << "max_receives=" << max_receives << "\n";

  ray::object_manager::test::ObjectManagerBenchmarkTool om(
      node_ip_address,
      redis_address,
      redis_port,
      max_sends,
      max_receives,
      mode,
      object_chunk_size,
      store_gigabytes_memory);

  om.ConnectAndExecute(mode, object_size, num_objects, num_trials, skip_k);
  om.main_service.run();
}
