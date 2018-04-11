#ifndef RAY_OBJECT_MANAGER_OBJECT_MANAGER_TEST_COMMON_H
#define RAY_OBJECT_MANAGER_OBJECT_MANAGER_TEST_COMMON_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <thread>

#include <boost/asio.hpp>
#include <boost/asio/error.hpp>
#include <boost/bind.hpp>

#include "ray/common/client_connection.h"
#include "ray/id.h"
#include "ray/status.h"

#include "plasma/client.h"
#include "plasma/events.h"
#include "plasma/plasma.h"

#include "ray/object_manager/object_manager.h"

namespace ray {
namespace object_manager {
namespace test {

std::pair<double_t,double_t> mean_std(const std::vector<double_t> &in_v, uint skip_n){
  std::vector<double_t> v;
  for (;skip_n<in_v.size();++skip_n) {
    v.push_back(in_v[skip_n]);
  }
  RAY_LOG(DEBUG) << "mean_std with n=" << v.size();
  double sum = std::accumulate(v.begin(), v.end(), 0.0);
  double mean = sum / v.size();
  double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / v.size() - mean * mean);
  return std::pair<double_t,double_t>(mean, stdev);
}

int64_t current_time_ms() {
  std::chrono::milliseconds ms_since_epoch =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch());
  return ms_since_epoch.count();
}

static inline void flushall_redis(void) {
  redisContext *context = redisConnect("127.0.0.1", 6379);
  freeReplyObject(redisCommand(context, "FLUSHALL"));
  redisFree(context);
}

std::string StartStore(const std::string &id, const std::string &store_executable, std::string gigabytes_memory) {
  std::string store_id = "/tmp/store";
  store_id = store_id + id;
  std::string store_pid = store_id + ".pid";
  std::string plasma_command = store_executable + " -m " + gigabytes_memory + "000000000 -s " + store_id +
      " 1> /dev/null 2> /dev/null &" + " echo $! > " +
      store_pid;
  RAY_LOG(DEBUG) << plasma_command;
  RAY_CHECK(!system(plasma_command.c_str()));
  sleep(1);
  return store_id;
}

void StopStore(std::string store_id) {
  std::string store_pid = store_id + ".pid";
  std::string kill_1 = "kill -9 `cat " + store_pid + "`";
  RAY_CHECK(!system(kill_1.c_str()));
}

ObjectID WriteDataToClient(plasma::PlasmaClient &client, uint64_t data_size, uint64_t metadata_size=1, bool randomize_data=false) {
  ObjectID object_id = ObjectID::from_random();
  RAY_LOG(DEBUG) << "ObjectID Created: " << object_id;
  uint8_t metadata[metadata_size];
  // Write random metadata.
  if (randomize_data) {
    srand(time(NULL));
    for (uint64_t i = 0; i < metadata_size; i++)
      metadata[i] = (uint8_t) (rand() % 256);
  }
  RAY_CHECK(sizeof(metadata) == metadata_size);
  std::shared_ptr<Buffer> data;
  ARROW_CHECK_OK(client.Create(object_id.to_plasma_id(), data_size, metadata,
                               metadata_size, &data));
  // Write random data to buffer.
  uint8_t *buffer = data->mutable_data();
  if (randomize_data) {
    srand(time(NULL));
    for (uint64_t i = 0; i < data_size; i++)
      buffer[i] = (uint8_t) (rand() % 256);
  }
  ARROW_CHECK_OK(client.Seal(object_id.to_plasma_id()));
  return object_id;
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
      *object_manager_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), 0)),
        gcs_client_(gcs_client),
        main_service_(main_service),
        object_manager_service_(object_manager_service.get()),
        object_manager_socket_(*object_manager_service),
        object_manager_(main_service, std::move(object_manager_service),
                        object_manager_config, gcs_client) {
    RAY_CHECK_OK(RegisterGcs(node_ip_address, redis_address, redis_port, main_service));
    // Start listening for clients.
    DoAcceptObjectManager();
  }

  ~MockServer() {
    object_manager_acceptor_.cancel();
    RAY_CHECK_OK(gcs_client_->client_table().Disconnect());
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

  friend class TestObjectManagerBasic;
  friend class StressTestObjectManager;
  friend class ObjectManagerBenchmarkTool;

  boost::asio::ip::tcp::acceptor object_manager_acceptor_;
  std::shared_ptr<gcs::AsyncGcsClient> gcs_client_;
  boost::asio::io_service &main_service_;
  boost::asio::io_service *object_manager_service_;
  boost::asio::ip::tcp::socket object_manager_socket_;
  ObjectManager object_manager_;
};

} // namespace test
} // namespace object_manager
} // namespace ray

#endif //RAY_OBJECT_MANAGER_OBJECT_MANAGER_TEST_COMMON_H
