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
                             int num_threads,
                             int max_sends,
                             int max_receives,
                             std::string mode,
                             uint64_t object_chunk_size,
                             const std::string &store_gigabytes_memory) {

    SetUp(node_ip_address,
          redis_address,
          redis_port,
          num_threads,
          max_sends,
          max_receives,
          mode,
          object_chunk_size,
          store_gigabytes_memory);
  }

  void SetUp(std::string node_ip_address,
             std::string redis_address,
             int redis_port,
             int num_threads,
             int max_sends,
             int max_receives,
             std::string mode,
             uint64_t object_chunk_size,
             const std::string &store_gigabytes_memory) {

    object_manager_service_1.reset(new boost::asio::io_service());
    work_.reset(new boost::asio::io_service::work(main_service));

    // start store
    std::string store_sock_1 = StartStore(mode, store_executable, store_gigabytes_memory);

    // start first server
    gcs_client_1 = std::shared_ptr<gcs::AsyncGcsClient>(new gcs::AsyncGcsClient());
    ObjectManagerConfig om_config_1;
    om_config_1.store_socket_name = store_sock_1;
    om_config_1.num_threads = num_threads;
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
    ARROW_CHECK_OK(client1.Connect(store_sock_1, "", PLASMA_DEFAULT_RELEASE_DELAY));
  }

  void TearDown() {
    main_service.post([this](){
      sleep(1);
      main_service.stop();
      ARROW_CHECK_OK(client1.Disconnect());
      this->server1.reset();
    });
    work_.reset();
  }

  void ConnectAndExecute(std::string mode, uint64_t object_size, int num_objects, int num_trials) {
    RAY_LOG(INFO) << "creating " << num_objects*num_trials
                  << " empty objects of size " << object_size;
    if (mode == "send"){
      // Create the objects to send before connecting.
      // The receiver will start timing as soon as the sender connects,
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
      // send a small object to initiate the send from sending side.
      init_object = test::WriteDataToClient(client1, 1, 0);
      status = server1->object_manager_.Push(init_object, remote_client_id);
      RAY_LOG(INFO) << "sent " << init_object;
      // start timer now since the sender will start sending as soon as it receives
      // the small object.
      status =
          server1->object_manager_.SubscribeObjAdded(
              [this, remote_client_id, object_size,
                  num_objects, num_trials](const RayObjectInfo &object_info) {
                if (init_object == object_info.object_id){
                  // ignore the initial object we sent out to start the experiment.
                  // start the timer here since we will certainly register object added
                  // before the remote object manager does.
                  start_time = current_time_ms();
                  return;
                }

                // record stats
                v1.push_back(object_info.object_id);
                receive_times.push_back(current_time_ms());

                if ((int)v1.size() == num_objects) {
                  for (uint i=0;i<v1.size();++i) {
                    RAY_LOG(DEBUG) << "received " << v1[i] << " " << receive_times[i];
                  }
                  double_t elapsed = current_time_ms() - start_time;
                  double_t gbits = (double)object_size*num_objects*8.0/1000.0/1000.0/1000.0;
                  double_t gbits_sec = gbits/(elapsed/1000.0);
                  int64_t min_time = *std::min_element(receive_times.begin(), receive_times.end());
                  int64_t max_time = *std::max_element(receive_times.begin(), receive_times.end());

                  elapsed_stats_.push_back(elapsed);
                  gbits_sec_stats_.push_back(gbits_sec);
                  duration_stats_.push_back((double)max_time-(double)min_time);

                  RAY_LOG(DEBUG) << "elapsed milliseconds " << elapsed;
                  RAY_LOG(DEBUG) << "GBits transferred " << gbits;
                  RAY_LOG(DEBUG) << "GBits/sec " << gbits_sec;
                  RAY_LOG(DEBUG) << "max=" << max_time << " min=" << min_time;
                  RAY_LOG(DEBUG) << "max-min time " << (max_time-min_time);

                  RAY_LOG(DEBUG) << "trial " << trial_count << " " << init_object;
                  trial_count += 1;
                  if (trial_count < num_trials) {
                    // clear stats
                    v1.clear();
                    receive_times.clear();
                    init_object = test::WriteDataToClient(client1, 1, 0);
                    Status push_status = server1->object_manager_.Push(init_object, remote_client_id);
                  } else {
                    std::pair<double_t,double_t> elapsed_stat = mean_std(elapsed_stats_, 3);
                    std::pair<double_t,double_t> gbits_sec_stat = mean_std(gbits_sec_stats_, 3);
                    std::pair<double_t,double_t> duration_stat = mean_std(duration_stats_, 3);

                    RAY_LOG(INFO) << "elapsed milliseconds "
                                  << "mean=" << elapsed_stat.first
                                  << " std=" << elapsed_stat.second;
                    RAY_LOG(INFO) << "GBits/sec "
                                  << "mean=" << gbits_sec_stat.first
                                  << " std=" << gbits_sec_stat.second;
                    RAY_LOG(INFO) << "max-min time "
                                  << "mean=" << duration_stat.first
                                  << " std=" << duration_stat.second;
                    TearDown();
                  }
                }
              }
          );
      RAY_CHECK_OK(status);
    } else if (mode == "send"){
      status =
          server1->object_manager_.SubscribeObjAdded(
              [this, remote_client_id, object_size, num_objects, num_trials](const RayObjectInfo &object_info) {
                if (ignore_send_ids.count(object_info.object_id) != 0) {
                  // send objects only when we receive an ObjectID we didn't send.
                  // this is the small object sent from the receiver.
                  return;
                }
                RAY_LOG(DEBUG) << "received " << object_info.object_id;
                start_time = current_time_ms();
                for (auto oid : send_object_ids[trial_count]) {
                  ray::Status async_status = server1->object_manager_.Push(oid, remote_client_id);
                  RAY_CHECK_OK(async_status);
                }
                int64_t elapsed = current_time_ms() - start_time;
                RAY_LOG(DEBUG) << "trial=" << trial_count
                               << " elapsed=" << elapsed
                               << " object_id=" << object_info.object_id;

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
    } else {
      RAY_LOG(FATAL) << mode << " is not a supported mode.";
    }
  }

  boost::asio::io_service main_service;

 protected:
  std::vector<std::unordered_set<ObjectID, UniqueIDHasher>> send_object_ids;
  std::unordered_set<ObjectID, UniqueIDHasher> ignore_send_ids;

  std::thread p;
  std::unique_ptr<boost::asio::io_service> object_manager_service_1;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_1;
  std::unique_ptr<test::MockServer> server1;

  plasma::PlasmaClient client1;
  std::vector<ObjectID> v1;
  std::vector<int64_t> receive_times;

  // experiment-specific variables
  int trial_count = 0;
  int64_t start_time;
  ObjectID init_object;

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

  uint64_t object_size = std::stol(argv[6]);
  int num_objects = std::stoi(argv[7]);
  int num_trials = std::stoi(argv[8]);

  int num_threads = std::stoi(argv[9]);
  int max_sends = std::stoi(argv[10]);
  int max_receives = std::stoi(argv[11]);
  uint64_t object_chunk_size = std::stol(argv[12]);
  const std::string store_gigabytes_memory = std::string(argv[13]);

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
      << "num_objects=         " << num_objects << "\n"
      << "object_size=         " << object_size << "\n"
      << "object_chunk_size=   " << object_chunk_size << "\n"
      << "num_chunks(computed)=" << num_chunks << "\n"

      << "\n"
      << " num_threads=" << num_threads << "\n"
      << "   max_sends=" << max_sends << "\n"
      << "max_receives=" << max_receives << "\n";

  ray::object_manager::test::ObjectManagerBenchmarkTool om(
      node_ip_address,
      redis_address,
      redis_port,
      num_threads,
      max_sends,
      max_receives,
      mode,
      object_chunk_size,
      store_gigabytes_memory);

  om.ConnectAndExecute(mode, object_size, num_objects, num_trials);
  om.main_service.run();
}
