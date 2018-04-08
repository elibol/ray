#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <numeric>

#include "ray/object_manager/object_manager.h"

namespace ray {

std::string store_executable;

int64_t current_time_ms() {
  std::chrono::milliseconds ms_since_epoch =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch());
  return ms_since_epoch.count();
}

class MockServer {
 public:
  MockServer(boost::asio::io_service &main_service,
             const std::string &node_ip_address,
             const std::string &redis_address,
             int redis_port,
             std::unique_ptr<boost::asio::io_service> object_manager_service,
             const ObjectManagerConfig &object_manager_config,
             std::shared_ptr<gcs::AsyncGcsClient> gcs_client)
      : object_manager_acceptor_(
      main_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0)),
        gcs_client_(gcs_client),
        main_service_(main_service),
        object_manager_service_(object_manager_service.get()),
        object_manager_socket_(main_service),
        object_manager_(main_service, std::move(object_manager_service),
                        object_manager_config, gcs_client) {
    RAY_CHECK_OK(RegisterGcs(node_ip_address, redis_address, redis_port, main_service));
    // Start listening for clients.
    DoAcceptObjectManager();
  }

  ~MockServer() {
    object_manager_acceptor_.cancel();
    RAY_CHECK_OK(gcs_client_->client_table().Disconnect());
    RAY_CHECK_OK(object_manager_.Terminate());
  }

 private:
  ray::Status RegisterGcs(const std::string &node_ip_address,
                          const std::string &redis_address, int redis_port,
                          boost::asio::io_service &io_service) {
    RAY_RETURN_NOT_OK(gcs_client_->Connect(redis_address, redis_port));
    RAY_RETURN_NOT_OK(gcs_client_->Attach(io_service));

    ClientTableDataT client_info = gcs_client_->client_table().GetLocalClient();
    client_info.node_manager_address = node_ip_address;
    client_info.object_manager_port = object_manager_acceptor_.local_endpoint().port();
    // Add resource information.

    RAY_LOG(DEBUG) << "Node manager listening on: IP " << client_info.node_manager_address
                   << " port " << client_info.node_manager_port;
    RAY_RETURN_NOT_OK(gcs_client_->client_table().Connect(client_info));

    auto node_manager_client_added = [this](gcs::AsyncGcsClient *client, const UniqueID &id,
                                            const ClientTableDataT &data) {
    };
    gcs_client_->client_table().RegisterClientAddedCallback(node_manager_client_added);
    return Status::OK();
  }

  void DoAcceptObjectManager() {
    object_manager_socket_ = boost::asio::ip::tcp::socket(*object_manager_service_);
    object_manager_acceptor_.async_accept(
        object_manager_socket_, boost::bind(&MockServer::HandleAcceptObjectManager, this,
                                            boost::asio::placeholders::error));
  }

  void HandleAcceptObjectManager(const boost::system::error_code &error) {
    ClientHandler<boost::asio::ip::tcp> client_handler =
        [this](std::shared_ptr<TcpClientConnection> client) {
          object_manager_.ProcessNewClient(client);
        };
    MessageHandler<boost::asio::ip::tcp> message_handler = [this](
        std::shared_ptr<TcpClientConnection> client, int64_t message_type,
        const uint8_t *message) {
      object_manager_.ProcessClientMessage(client, message_type, message);
    };
    // Accept a new local client and dispatch it to the node manager.
    auto new_connection = TcpClientConnection::Create(client_handler, message_handler,
                                                      std::move(object_manager_socket_));
    DoAcceptObjectManager();
  }

  friend class MultinodeObjectManagerTest;

  boost::asio::ip::tcp::acceptor object_manager_acceptor_;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_;
  boost::asio::io_service &main_service_;
  boost::asio::io_service *object_manager_service_;
  boost::asio::ip::tcp::socket object_manager_socket_;
  ObjectManager object_manager_;
};

class MultinodeObjectManagerTest {

 public:

  MultinodeObjectManagerTest(std::string node_ip_address,
                             std::string redis_address,
                             int redis_port,
                             int num_threads,
                             int max_sends,
                             int max_receives,
                             std::string mode) {
    SetUp(node_ip_address,
          redis_address,
          redis_port,
          num_threads,
          max_sends,
          max_receives,
          mode);
  }

  std::string StartStore(const std::string &id) {
    std::string store_id = "/tmp/store";
    store_id = store_id + id;
    std::string plasma_command = store_executable + " -m 16000000000 -s " + store_id +
        " 1> /dev/null 2> /dev/null &";
    RAY_LOG(DEBUG) << plasma_command;
    int ec = system(plasma_command.c_str());
    if (ec != 0) {
      throw std::runtime_error("failed to start plasma store.");
    };
    sleep(1);
    return store_id;
  }

  void SetUp(std::string node_ip_address,
             std::string redis_address,
             int redis_port,
             int num_threads,
             int max_sends,
             int max_receives,
             std::string mode) {

    object_manager_service_1.reset(new boost::asio::io_service());

    // start store
    std::string store_sock_1 = StartStore(mode);

    // start first server
    gcs_client_1 = std::shared_ptr<gcs::AsyncGcsClient>(new gcs::AsyncGcsClient());
    ObjectManagerConfig om_config_1;
    om_config_1.store_socket_name = store_sock_1;
    // good enough settings for m4.16xlarge
    om_config_1.num_threads = num_threads;
    om_config_1.max_sends = max_sends;
    om_config_1.max_receives = max_receives;
    server1.reset(new MockServer(main_service,
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
    main_service.stop();
    arrow::Status client1_status = client1.Disconnect();
    RAY_CHECK(client1_status.ok());
    this->server1.reset();
    int s = system("killall plasma_store &");
    RAY_CHECK(!s);
  }

  ObjectID WriteDataToClient(plasma::PlasmaClient &client, int64_t data_size) {
    ObjectID object_id = ObjectID::from_random();
    RAY_LOG(DEBUG) << "ObjectID Created: " << object_id;
    uint8_t metadata[] = {5};
    int64_t metadata_size = sizeof(metadata);
    std::shared_ptr<Buffer> data;
    ARROW_CHECK_OK(client.Create(object_id.to_plasma_id(), data_size, metadata,
                                 metadata_size, &data));
    ARROW_CHECK_OK(client.Seal(object_id.to_plasma_id()));
    return object_id;
  }

  void ConnectAndExecute(std::string mode, int object_size, int num_objects, int num_trials) {
    if (mode == "send"){
      // Create the objects to send before connecting.
      // The receiver will start timing as soon as the sender connects,
      // so we want to make sure we're not timing object creation.
      for (int trial=0; trial < num_trials; ++trial){
        send_object_ids.emplace_back(std::unordered_set<ObjectID, UniqueIDHasher>());
        for (int i=0;i<num_objects;++i) {
          ObjectID oid = WriteDataToClient(client1, object_size);
          send_object_ids[trial].insert(oid);
          ignore_send_ids.insert(oid);
        }
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

  void Execute(ClientID remote_client_id, std::string mode, int object_size, int num_objects, int num_trials){
    RAY_LOG(INFO) << "remote client_id " << remote_client_id;
    ray::Status status = ray::Status::OK();
    if (mode == "receive"){
      // send a small object to initiate the send from sending side.
      init_object = WriteDataToClient(client1, 1);
      status = server1->object_manager_.Push(init_object, remote_client_id);
      RAY_LOG(INFO) << "sent " << init_object;
      // start timer now since the sender will start sending as soon as it receives
      // the small object.
      status =
          server1->object_manager_.SubscribeObjAdded(
              [this, remote_client_id, object_size,
                  num_objects, num_trials](const ObjectID &object_id) {
                if (init_object == object_id){
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
                  for (uint i=0;i<v1.size();++i) {
                    RAY_LOG(INFO) << "received " << v1[i] << " " << receive_times[i];
                  }
                  double_t elapsed = current_time_ms() - start_time;
                  double_t gbits = (double)object_size*num_objects*8.0/1000.0/1000.0/1000.0;
                  double_t gbits_sec = gbits/(elapsed/1000.0);
                  int64_t min_time = *std::min_element(receive_times.begin(), receive_times.end());
                  int64_t max_time = *std::max_element(receive_times.begin(), receive_times.end());

                  elapsed_stats_.push_back(elapsed);
                  gbits_sec_stats_.push_back(gbits_sec);
                  duration_stats_.push_back((double)max_time-(double)min_time);

                  RAY_LOG(INFO) << "elapsed milliseconds " << elapsed;
                  RAY_LOG(INFO) << "GBits transferred " << gbits;
                  RAY_LOG(INFO) << "GBits/sec " << gbits_sec;
                  RAY_LOG(INFO) << "max=" << max_time << " min=" << min_time;
                  RAY_LOG(INFO) << "max-min time " << (max_time-min_time);

                  trial_count += 1;
                  if (trial_count < num_trials) {
                    // clear stats
                    v1.clear();
                    receive_times.clear();
                    init_object = WriteDataToClient(client1, 1);
                    Status push_status = server1->object_manager_.Push(init_object, remote_client_id);
                    RAY_LOG(INFO) << "sent " << init_object;
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
                  }
                  TearDown();
                }
              }
          );
      RAY_CHECK_OK(status);
    } else if (mode == "send"){
      status =
          server1->object_manager_.SubscribeObjAdded(
              [this, remote_client_id, object_size, num_objects, num_trials](const ObjectID &incoming_object_id) {
                if (ignore_send_ids.count(incoming_object_id) != 0) {
                  // send objects only when we receive an ObjectID we didn't send.
                  // this is the small object sent from the receiver.
                  return;
                }
                RAY_LOG(INFO) << "received " << incoming_object_id;
                start_time = current_time_ms();
                for (auto oid : send_object_ids[trial_count]) {
                  ray::Status async_status = server1->object_manager_.Push(oid, remote_client_id);
                  RAY_CHECK_OK(async_status);
                }
                int64_t elapsed = current_time_ms() - start_time;
                RAY_LOG(INFO) << "elapsed " << elapsed;

                for (auto oid : send_object_ids[trial_count]) {
                  RAY_LOG(INFO) << "sent " << oid;
                }
                trial_count += 1;
                if (trial_count >= num_trials){
                  // TearDown();
                }
              }
          );
      RAY_CHECK_OK(status);
    } else {
      RAY_LOG(FATAL) << mode << " is not a supported mode.";
    }
  }

  std::pair<double_t,double_t> mean_std(const std::vector<double_t> &in_v, uint skip_n){
    std::vector<double_t> v;
    for (;skip_n<in_v.size();++skip_n) {
      v.push_back(in_v[skip_n]);
    }
    RAY_LOG(INFO) << "mean_std with n=" << v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    return std::pair<double_t,double_t>(mean, stdev);
  }

  boost::asio::io_service main_service;

 protected:
  std::vector<std::unordered_set<ObjectID, UniqueIDHasher>> send_object_ids;
  std::unordered_set<ObjectID, UniqueIDHasher> ignore_send_ids;

  std::thread p;
  std::unique_ptr<boost::asio::io_service> object_manager_service_1;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_1;
  std::unique_ptr<MockServer> server1;

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
};


}  // namespace ray

int main(int argc, char **argv) {
  const std::string node_ip_address = std::string(argv[1]);
  const std::string redis_address = std::string(argv[2]);
  int redis_port = std::stoi(argv[3]);
  ray::store_executable = std::string(argv[4]);
  const std::string mode = std::string(argv[5]);

  int object_size = std::stoi(argv[6]);
  int num_objects = std::stoi(argv[7]);
  int num_trials = std::stoi(argv[8]);

  int num_threads = std::stoi(argv[9]);
  int max_sends = std::stoi(argv[10]);
  int max_receives = std::stoi(argv[11]);

  RAY_LOG(INFO) <<"\n"
      << "node_ip_address=" << node_ip_address << "\n"
      << "redis_address=" << redis_address << "\n"
      << "redis_port=" << redis_port << "\n"
      << "store_executable=" << store_executable << "\n"

      << "\n"
      << "mode=" << mode << "\n"
      << "object_size=" << object_size << "\n"
      << "num_objects=" << num_objects << "\n"
      << "num_trials=" << num_trials << "\n"

      << "\n"
      << "num_threads=" << num_threads << "\n"
      << "max_sends=" << max_sends << "\n"
      << "max_receives=" << max_receives << "\n";

  MultinodeObjectManagerTest om(node_ip_address,
                                redis_address,
                                redis_port,
                                num_threads,
                                max_sends,
                                max_receives,
                                mode);

  om.ConnectAndExecute(mode, object_size, num_objects, num_trials);
  om.main_service.run();
}
