#include <chrono>
#include <iostream>
#include <random>
#include <thread>

#include "gtest/gtest.h"

#include "ray/object_manager/object_manager.h"
#include "ray/object_manager/test/object_manager_test_common.h"

namespace ray {
namespace object_manager {
namespace test {

std::string store_executable;

class StressTestObjectManagerBase : public ::testing::Test {
 public:
  StressTestObjectManagerBase() {}

  void SetUp() {
    test::flushall_redis();

    object_manager_service_1.reset(new boost::asio::io_service());
    object_manager_service_2.reset(new boost::asio::io_service());

    // start store
    store_id_1 = StartStore(UniqueID::from_random().hex(), store_executable, "1");
    store_id_2 = StartStore(UniqueID::from_random().hex(), store_executable, "1");

    uint pull_timeout_ms = 1;
    int max_sends = 2;
    int max_receives = 2;
    uint64_t object_chunk_size = static_cast<uint64_t>(std::pow(10, 3));

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

class StressTestObjectManager : public StressTestObjectManagerBase {
 public:
  enum TransferPattern {
    PUSH_A_B,
    PUSH_B_A,
    BIDIRECTIONAL_PUSH,
    PULL_A_B,
    PULL_B_A,
    BIDIRECTIONAL_PULL,
    BIDIRECTIONAL_PULL_VARIABLE_DATA_SIZE,
  };

  int async_loop_index = -1;
  uint num_expected_objects;

  std::vector<TransferPattern> async_loop_patterns = {
      PUSH_A_B,
      PUSH_B_A,
      BIDIRECTIONAL_PUSH,
      PULL_A_B,
      PULL_B_A,
      BIDIRECTIONAL_PULL,
      BIDIRECTIONAL_PULL_VARIABLE_DATA_SIZE};

  int num_connected_clients = 0;

  ClientID client_id_1;
  ClientID client_id_2;

  int64_t start_time;

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
    AddTransferTestHandlers();
    TransferTestNext();
  }

  void AddTransferTestHandlers() {
    ray::Status status = ray::Status::OK();
    status = server1->object_manager_.SubscribeObjAdded(
        [this](const ObjectInfoT &object_info) {
          object_added_handler_1(ObjectID::from_binary(object_info.object_id));
          if (v1.size() == num_expected_objects && v1.size() == v2.size()) {
            TransferTestComplete();
          }
        });
    RAY_CHECK_OK(status);
    status = server2->object_manager_.SubscribeObjAdded(
        [this](const ObjectInfoT &object_info) {
          object_added_handler_2(ObjectID::from_binary(object_info.object_id));
          if (v2.size() == num_expected_objects && v1.size() == v2.size()) {
            TransferTestComplete();
          }
        });
    RAY_CHECK_OK(status);
  }

  void TransferTestNext() {
    async_loop_index += 1;
    if ((uint)async_loop_index < async_loop_patterns.size()) {
      TransferPattern pattern = async_loop_patterns[async_loop_index];
      TransferTestExecute(100, 3 * std::pow(10, 4) - 1, pattern);
    } else {
      main_service.stop();
    }
  }

  plasma::ObjectBuffer GetObject(plasma::PlasmaClient &client, ObjectID &object_id) {
    plasma::ObjectBuffer object_buffer;
    plasma::ObjectID plasma_id = object_id.to_plasma_id();
    ARROW_CHECK_OK(client.Get(&plasma_id, 1, 0, &object_buffer));
    return object_buffer;
  }

  static unsigned char *GetDigest(plasma::PlasmaClient &client, ObjectID &object_id) {
    const int64_t size = sizeof(uint64_t);
    static unsigned char digest_1[size];
    ARROW_CHECK_OK(client.Hash(object_id.to_plasma_id(), &digest_1[0]));
    return digest_1;
  }

  void CompareObjects(ObjectID &object_id_1, ObjectID &object_id_2) {
    plasma::ObjectBuffer object_buffer_1 = GetObject(client1, object_id_1);
    plasma::ObjectBuffer object_buffer_2 = GetObject(client2, object_id_2);
    uint8_t *data_1 = const_cast<uint8_t *>(object_buffer_1.data->data());
    uint8_t *data_2 = const_cast<uint8_t *>(object_buffer_2.data->data());
    ASSERT_EQ(object_buffer_1.data_size, object_buffer_2.data_size);
    ASSERT_EQ(object_buffer_1.metadata_size, object_buffer_2.metadata_size);
    int64_t total_size = object_buffer_1.data_size + object_buffer_1.metadata_size;
    RAY_LOG(DEBUG) << "total_size " << total_size;
    for (int i = -1; ++i < total_size;) {
      ASSERT_TRUE(data_1[i] == data_2[i]);
    }
  }

  void CompareHashes(ObjectID &object_id_1, ObjectID &object_id_2) {
    const int64_t size = sizeof(uint64_t);
    static unsigned char *digest_1 = GetDigest(client1, object_id_1);
    static unsigned char *digest_2 = GetDigest(client2, object_id_2);
    for (int i = -1; ++i < size;) {
      ASSERT_TRUE(digest_1[i] == digest_2[i]);
    }
  }

  void TransferTestComplete() {
    int64_t elapsed = current_time_ms() - start_time;
    RAY_LOG(INFO) << "TransferTestComplete: " << async_loop_patterns[async_loop_index]
                  << " " << v1.size() << " " << elapsed;
    ASSERT_TRUE(v1.size() == v2.size());
    for (uint i = 0; i < v1.size(); ++i) {
      ASSERT_TRUE(std::find(v1.begin(), v1.end(), v2[i]) != v1.end());
    }

    // Compare objects and their hashes.
    for (uint i = 0; i < v1.size(); ++i) {
      ObjectID object_id_2 = v2[i];
      ObjectID object_id_1 =
          v1[std::distance(v1.begin(), std::find(v1.begin(), v1.end(), v2[i]))];
      CompareHashes(object_id_1, object_id_2);
      CompareObjects(object_id_1, object_id_2);
    }

    v1.clear();
    v2.clear();
    TransferTestNext();
  }

  void TransferTestExecute(int num_trials, int64_t data_size,
                           TransferPattern transfer_pattern) {
    ClientID client_id_1 = gcs_client_1->client_table().GetLocalClientId();
    ClientID client_id_2 = gcs_client_2->client_table().GetLocalClientId();

    ray::Status status = ray::Status::OK();

    if (transfer_pattern == BIDIRECTIONAL_PULL ||
        transfer_pattern == BIDIRECTIONAL_PUSH ||
        transfer_pattern == BIDIRECTIONAL_PULL_VARIABLE_DATA_SIZE) {
      num_expected_objects = (uint)2 * num_trials;
    } else {
      num_expected_objects = (uint)num_trials;
    }

    start_time = current_time_ms();

    switch (transfer_pattern) {
    case PUSH_A_B: {
      for (int i = -1; ++i < num_trials;) {
        ObjectID oid1 = test::WriteDataToClient(client1, data_size, 10, true);
        status = server1->object_manager_.Push(oid1, client_id_2);
      }
    } break;
    case PUSH_B_A: {
      for (int i = -1; ++i < num_trials;) {
        ObjectID oid2 = test::WriteDataToClient(client2, data_size, 10, true);
        status = server2->object_manager_.Push(oid2, client_id_1);
      }
    } break;
    case BIDIRECTIONAL_PUSH: {
      for (int i = -1; ++i < num_trials;) {
        ObjectID oid1 = test::WriteDataToClient(client1, data_size, 10, true);
        status = server1->object_manager_.Push(oid1, client_id_2);
        ObjectID oid2 = test::WriteDataToClient(client2, data_size, 10, true);
        status = server2->object_manager_.Push(oid2, client_id_1);
      }
    } break;
    case PULL_A_B: {
      for (int i = -1; ++i < num_trials;) {
        ObjectID oid1 = test::WriteDataToClient(client1, data_size, 10, true);
        status = server2->object_manager_.Pull(oid1);
      }
    } break;
    case PULL_B_A: {
      for (int i = -1; ++i < num_trials;) {
        ObjectID oid2 = test::WriteDataToClient(client2, data_size, 10, true);
        status = server1->object_manager_.Pull(oid2);
      }
    } break;
    case BIDIRECTIONAL_PULL: {
      for (int i = -1; ++i < num_trials;) {
        ObjectID oid1 = test::WriteDataToClient(client1, data_size, 10, true);
        status = server2->object_manager_.Pull(oid1);
        ObjectID oid2 = test::WriteDataToClient(client2, data_size, 10, true);
        status = server1->object_manager_.Pull(oid2);
      }
    } break;
    case BIDIRECTIONAL_PULL_VARIABLE_DATA_SIZE: {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis(1, 50);
      for (int i = -1; ++i < num_trials;) {
        ObjectID oid1 = test::WriteDataToClient(client1, data_size + dis(gen), dis(gen), true);
        status = server2->object_manager_.Pull(oid1);
        ObjectID oid2 = test::WriteDataToClient(client2, data_size + dis(gen), dis(gen), true);
        status = server1->object_manager_.Pull(oid2);
      }
    } break;
    default: {
      RAY_LOG(FATAL) << "No case for transfer_pattern " << transfer_pattern;
    } break;
    }
  }

  void TestConnections() {
    RAY_LOG(DEBUG) << "\n"
                   << "Server client ids:"
                   << "\n";
    ClientID client_id_1 = gcs_client_1->client_table().GetLocalClientId();
    ClientID client_id_2 = gcs_client_2->client_table().GetLocalClientId();
    RAY_LOG(DEBUG) << "Server 1: " << client_id_1 << "\n"
                   << "Server 2: " << client_id_2;

    RAY_LOG(DEBUG) << "\n"
                   << "All connected clients:"
                   << "\n";
    const ClientTableDataT &data = gcs_client_1->client_table().GetClient(client_id_1);
    RAY_LOG(DEBUG) << "ClientID=" << ClientID::from_binary(data.client_id) << "\n"
                   << "ClientIp=" << data.node_manager_address << "\n"
                   << "ClientPort=" << data.node_manager_port;
    const ClientTableDataT &data2 = gcs_client_1->client_table().GetClient(client_id_2);
    RAY_LOG(DEBUG) << "ClientID=" << ClientID::from_binary(data2.client_id) << "\n"
                   << "ClientIp=" << data2.node_manager_address << "\n"
                   << "ClientPort=" << data2.node_manager_port;
  }
};

TEST_F(StressTestObjectManager, StartStressTestObjectManager) {
  auto AsyncStartTests = main_service.wrap([this]() { WaitConnections(); });
  AsyncStartTests();
  main_service.run();
}

} // namespace test
} // namespace object_manager
}  // namespace ray

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ray::object_manager::test::store_executable = std::string(argv[1]);
  return RUN_ALL_TESTS();
}
