#include <iostream>
#include <thread>

#include "gtest/gtest.h"

#include "ray/object_manager/object_manager.h"
#include "ray/object_manager/test/object_manager_test_common.h"

namespace ray {
namespace object_manager {
namespace test {

std::string store_executable;

class TestObjectManager : public ::testing::Test {
 public:
  TestObjectManager() {}

  std::string StartStore(const std::string &id) {
    std::string store_id = "/tmp/store";
    store_id = store_id + id;
    std::string store_pid = store_id + ".pid";
    std::string plasma_command = store_executable + " -m 4000000000 -s " + store_id +
                                 " 1> /dev/null 2> /dev/null &" + " echo $! > " +
                                 store_pid;

    RAY_LOG(DEBUG) << plasma_command;
    int ec = system(plasma_command.c_str());
    RAY_CHECK(ec == 0);
    sleep(1);
    return store_id;
  }

  void SetUp() {
    test::flushall_redis();

    object_manager_service_1.reset(new boost::asio::io_service());
    object_manager_service_2.reset(new boost::asio::io_service());

    // start store
    store_id_1 = StartStore(UniqueID::from_random().hex());
    store_id_2 = StartStore(UniqueID::from_random().hex());

    uint pull_timeout_ms = 1;
    int max_sends = 2;
    int max_receives = 2;
    uint64_t object_chunk_size = static_cast<uint64_t>(std::pow(10, 10));

    // start first server
    gcs_client_1 = std::shared_ptr<gcs::AsyncGcsClient>(new gcs::AsyncGcsClient());
    ObjectManagerConfig om_config_1;
    om_config_1.store_socket_name = store_id_1;
    om_config_1.pull_timeout_ms = pull_timeout_ms;
    om_config_1.max_sends = max_sends;
    om_config_1.max_receives = max_receives;
    om_config_1.object_chunk_size = object_chunk_size;
    server1.reset(new test::MockServer(main_service,
                                       "127.0.0.1",
                                       "127.0.0.1",
                                       6379,
                                       std::move(object_manager_service_1),
                                       om_config_1, gcs_client_1));

    // start second server
    gcs_client_2 = std::shared_ptr<gcs::AsyncGcsClient>(new gcs::AsyncGcsClient());
    ObjectManagerConfig om_config_2;
    om_config_2.store_socket_name = store_id_2;
    om_config_2.pull_timeout_ms = pull_timeout_ms;
    om_config_2.max_sends = max_sends;
    om_config_2.max_receives = max_receives;
    om_config_2.object_chunk_size = object_chunk_size;
    server2.reset(new test::MockServer(main_service,
                                       "127.0.0.1",
                                       "127.0.0.1",
                                       6379,
                                       std::move(object_manager_service_2),
                                       om_config_2, gcs_client_2));

    // connect to stores.
    ARROW_CHECK_OK(client1.Connect(store_id_1, "", PLASMA_DEFAULT_RELEASE_DELAY));
    ARROW_CHECK_OK(client2.Connect(store_id_2, "", PLASMA_DEFAULT_RELEASE_DELAY));
  }

  void TearDown() {
    arrow::Status client1_status = client1.Disconnect();
    arrow::Status client2_status = client2.Disconnect();
    ASSERT_TRUE(client1_status.ok() && client2_status.ok());

    this->server1.reset();
    this->server2.reset();

    StopStore(store_id_1);
    StopStore(store_id_2);
  }

  void object_added_handler_1(ObjectID object_id) { v1.push_back(object_id); };

  void object_added_handler_2(ObjectID object_id) { v2.push_back(object_id); };

 protected:
  std::thread p;
  boost::asio::io_service main_service;
  std::unique_ptr<boost::asio::io_service> object_manager_service_1;
  std::unique_ptr<boost::asio::io_service> object_manager_service_2;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_1;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_2;
  std::unique_ptr<test::MockServer> server1;
  std::unique_ptr<test::MockServer> server2;

  plasma::PlasmaClient client1;
  plasma::PlasmaClient client2;
  std::vector<ObjectID> v1;
  std::vector<ObjectID> v2;

  std::string store_id_1;
  std::string store_id_2;
};

class TestObjectManagerBasic : public TestObjectManager {
 public:
  int num_connected_clients = 0;
  uint num_expected_objects;
  ClientID client_id_1;
  ClientID client_id_2;

  ObjectID created_object_id;
  enum class TestMode : int { NOTIFICATION = 0, TRANSFER };

  TestMode current_test_mode;

  void WaitConnections() {
    client_id_1 = gcs_client_1->client_table().GetLocalClientId();
    client_id_2 = gcs_client_2->client_table().GetLocalClientId();
    gcs_client_1->client_table().RegisterClientAddedCallback([this](
        gcs::AsyncGcsClient *client, const ClientID &id, const ClientTableDataT &data) {
      ClientID parsed_id = ClientID::from_binary(data.client_id);
      if (parsed_id == client_id_1 || parsed_id == client_id_2) {
        num_connected_clients += 1;
      }
      if (num_connected_clients == 2) {
        StartTests();
      }
    });
  }

  void StartTests() {
    TestConnections();
    TestNotifications();
  }

  void TestNotifications() {
    current_test_mode = TestMode::NOTIFICATION;
    ray::Status status = ray::Status::OK();
    status = server1->object_manager_.SubscribeObjAdded(
        [this](const ObjectInfoT &object_info) {
          if (current_test_mode != TestMode::NOTIFICATION) {
            return;
          }
          object_added_handler_1(ObjectID::from_binary(object_info.object_id));
          if (v1.size() == num_expected_objects) {
            NotificationTestComplete(created_object_id,
                                     ObjectID::from_binary(object_info.object_id));
          }
        });
    RAY_CHECK_OK(status);

    num_expected_objects = 1;
    uint64_t data_size = static_cast<uint64_t>(std::pow(10, 9));
    created_object_id = WriteDataToClient(client1, data_size);
  }

  void NotificationTestComplete(ObjectID object_id_1, ObjectID object_id_2) {
    ASSERT_EQ(object_id_1, object_id_2);
    v1.clear();
    TestLargeObjectTransfer();
  }

  void TestLargeObjectTransfer() {
    current_test_mode = TestMode::TRANSFER;
    ray::Status status = ray::Status::OK();
    status = server2->object_manager_.SubscribeObjAdded(
        [this](const ObjectInfoT &object_info) {
          if (current_test_mode != TestMode::TRANSFER) {
            return;
          }
          ObjectID remote_object_id = ObjectID::from_binary(object_info.object_id);
          ASSERT_EQ(created_object_id, remote_object_id);
          CompareHashes(client1, client2, created_object_id, remote_object_id);
          CompareObjects(client1, client2, created_object_id, remote_object_id);
          TestLargeObjectTransferComplete();
        });
    num_expected_objects = 1;
    uint64_t data_size = static_cast<uint64_t>(std::pow(10, 9));
    uint64_t metadata_size = static_cast<uint64_t>(std::pow(10, 6));
    created_object_id = WriteDataToClient(client1, data_size, metadata_size, true);
    RAY_CHECK_OK(server2->object_manager_.Pull(created_object_id));
  }

  void TestLargeObjectTransferComplete() { main_service.stop(); }

  void TestConnections() {
    RAY_LOG(DEBUG) << "\n"
                   << "Server client ids:"
                   << "\n";
    const ClientTableDataT &data = gcs_client_1->client_table().GetClient(client_id_1);
    RAY_LOG(DEBUG) << (ClientID::from_binary(data.client_id) == ClientID::nil());
    RAY_LOG(DEBUG) << "Server 1 ClientID=" << ClientID::from_binary(data.client_id);
    RAY_LOG(DEBUG) << "Server 1 ClientIp=" << data.node_manager_address;
    RAY_LOG(DEBUG) << "Server 1 ClientPort=" << data.node_manager_port;
    ASSERT_EQ(client_id_1, ClientID::from_binary(data.client_id));
    const ClientTableDataT &data2 = gcs_client_1->client_table().GetClient(client_id_2);
    RAY_LOG(DEBUG) << "Server 2 ClientID=" << ClientID::from_binary(data2.client_id);
    RAY_LOG(DEBUG) << "Server 2 ClientIp=" << data2.node_manager_address;
    RAY_LOG(DEBUG) << "Server 2 ClientPort=" << data2.node_manager_port;
    ASSERT_EQ(client_id_2, ClientID::from_binary(data2.client_id));
  }
};

TEST_F(TestObjectManagerBasic, StartTestObjectManagerBasic) {
  auto AsyncStartTests = main_service.wrap([this]() { WaitConnections(); });
  AsyncStartTests();
  main_service.run();
}

} // namespace test
} // namespace object_manager
} // namespace ray

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ray::object_manager::test::store_executable = std::string(argv[1]);
  return RUN_ALL_TESTS();
}
