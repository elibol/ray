#include <chrono>
#include <iostream>
#include <random>
#include <thread>

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
        object_manager_socket_(main_service),
        gcs_client_(gcs_client),
        object_manager_(main_service, std::move(object_manager_service),
                        object_manager_config, gcs_client) {
    RAY_CHECK_OK(RegisterGcs(node_ip_address, redis_address, redis_port, main_service));
    // Start listening for clients.
    DoAcceptObjectManager();
  }

  ~MockServer() {
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
  boost::asio::ip::tcp::socket object_manager_socket_;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_;
  ObjectManager object_manager_;
};

class MultinodeObjectManagerTest {

 public:

  MultinodeObjectManagerTest(std::string node_ip_address,
                             std::string redis_address,
                             int redis_port) {
    SetUp(node_ip_address,
          redis_address,
          redis_port);
  }

  std::string StartStore(const std::string &id) {
    std::string store_id = "/tmp/store";
    store_id = store_id + id;
    std::string plasma_command = store_executable + " -m 20000000000 -s " + store_id +
        " 1> /dev/null 2> /dev/null &";
    RAY_LOG(DEBUG) << plasma_command;
    int ec = system(plasma_command.c_str());
    if (ec != 0) {
      throw std::runtime_error("failed to start plasma store.");
    };
    return store_id;
  }

  void SetUp(std::string node_ip_address,
             std::string redis_address,
             int redis_port) {

    object_manager_service_1.reset(new boost::asio::io_service());

    // start store
    std::string store_sock_1 = StartStore("1");

    // start first server
    gcs_client_1 = std::shared_ptr<gcs::AsyncGcsClient>(new gcs::AsyncGcsClient());
    ObjectManagerConfig om_config_1;
    om_config_1.store_socket_name = store_sock_1;
    // good enough settings for m4.16xlarge
    om_config_1.num_threads = 32;
    om_config_1.max_sends = 16;
    om_config_1.max_receives = 16;
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

  void ConnectAndExecute(std::string mode, int object_size, int num_objects) {
    if (mode == "send"){
      // Create the objects to send before connecting.
      // The receiver will start timing as soon as the sender connects,
      // so we want to make sure we're not timing object creation.
      for (int i=0;i<num_objects;++i) {
        ObjectID oid = WriteDataToClient(client1, object_size);
        send_object_ids.insert(oid);
      }
    }
    ClientID client_id_1 = gcs_client_1->client_table().GetLocalClientId();
    RAY_LOG(INFO) << "local client_id " << client_id_1;
    gcs_client_1->client_table().RegisterClientAddedCallback(
        [this, client_id_1, mode, object_size, num_objects]
            (gcs::AsyncGcsClient *client, const ClientID &id, const ClientTableDataT &data) {
          ClientID parsed_id = ClientID::from_binary(data.client_id);
          if (!(parsed_id == client_id_1)){
            Execute(parsed_id, mode, object_size, num_objects);
          }
    });
  }

  void Execute(ClientID remote_client_id, std::string mode, int object_size, int num_objects){
    RAY_LOG(INFO) << "remote client_id " << remote_client_id;
    ray::Status status = ray::Status::OK();
    if (mode == "receive"){
      // send a small object to initiate the send from sending side.
      ObjectID init_object = WriteDataToClient(client1, 1);
      status = server1->object_manager_.Push(init_object, remote_client_id);
      RAY_LOG(INFO) << "sent " << init_object;
      // start timer now since the sender will start sending as soon as it receives
      // the small object.
      int64_t start_time = current_time_ms();
      status =
          server1->object_manager_.SubscribeObjAdded(
              [this, init_object, start_time, object_size, num_objects](const ObjectID &object_id) {
                if (init_object == object_id){
                  // ignore the initial object we sent out to start the experiment.
                  return;
                }
                v1.push_back(object_id);
                if ((int)v1.size() == num_objects) {
                  double_t elapsed = current_time_ms() - start_time;
                  double_t gbits = (double)object_size*num_objects*8.0/1000.0/1000.0/1000.0;
                  // double_t throughput = float(object_size*num_objects)*8.0/1000.0/1000.0/1000.0/(elapsed/1000.0);
                  double_t gbits_sec = gbits/(elapsed/1000.0);
                  RAY_LOG(INFO) << "elapsed milliseconds " << elapsed;
                  RAY_LOG(INFO) << "GBits transferred " << gbits;
                  RAY_LOG(INFO) << "GBits/sec " << gbits_sec;

                  for (auto v1oid : v1) {
                    RAY_LOG(INFO) << "received " << v1oid;
                  }

                  TearDown();
                }
              }
          );
      RAY_CHECK_OK(status);
    } else if (mode == "send"){
      status =
          server1->object_manager_.SubscribeObjAdded(
              [this, remote_client_id, object_size, num_objects](const ObjectID &incoming_object_id) {
                if (send_object_ids.count(incoming_object_id) != 0) {
                  // start when we receive an ObjectID we didn't send.
                  // this is the small object sent from the receiver.
                  return;
                }
                RAY_LOG(INFO) << "received " << incoming_object_id;
                int64_t start_time = current_time_ms();
                for (auto oid : send_object_ids) {
                  ray::Status async_status = server1->object_manager_.Push(oid, remote_client_id);
                  RAY_CHECK_OK(async_status);
                }
                int64_t elapsed = current_time_ms() - start_time;
                RAY_LOG(INFO) << "elapsed " << elapsed;
                for (auto oid : send_object_ids) {
                  RAY_LOG(INFO) << "sent " << oid;
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
  std::unordered_set<ObjectID, UniqueIDHasher> send_object_ids;

  std::thread p;
  std::unique_ptr<boost::asio::io_service> object_manager_service_1;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_1;
  std::unique_ptr<MockServer> server1;

  plasma::PlasmaClient client1;
  std::vector<ObjectID> v1;
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

  RAY_LOG(INFO) <<"\n"
      << "node_ip_address=" << node_ip_address << "\n"
      << "redis_address=" << redis_address << "\n"
      << "redis_port=" << redis_port << "\n"
      << "store_executable=" << store_executable << "\n"
      << "mode=" << mode << "\n"
      << "object_size=" << object_size << "\n"
      << "num_objects=" << num_objects << "\n";

  MultinodeObjectManagerTest om(node_ip_address,
                                redis_address,
                                redis_port);

  om.ConnectAndExecute(mode, object_size, num_objects);
  om.main_service.run();
}
