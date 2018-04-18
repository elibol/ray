#ifndef RAY_OBJECT_MANAGER_OBJECT_MANAGER_H
#define RAY_OBJECT_MANAGER_OBJECT_MANAGER_H

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <thread>

#include <boost/asio.hpp>
#include <boost/asio/error.hpp>
#include <boost/bind.hpp>

#include "plasma/client.h"
#include "plasma/events.h"
#include "plasma/plasma.h"

#include "ray/common/client_connection.h"
#include "ray/id.h"
#include "ray/status.h"

#include "ray/object_manager/connection_pool.h"
#include "ray/object_manager/format/object_manager_generated.h"
#include "ray/object_manager/object_buffer_pool.h"
#include "ray/object_manager/object_directory.h"
#include "ray/object_manager/object_manager_client_connection.h"
#include "ray/object_manager/object_store_notification_manager.h"

namespace ray {

struct ObjectManagerConfig {
  /// The time in milliseconds to wait before retrying a pull
  /// that failed due to client id lookup.
  uint pull_timeout_ms;
  /// Maximum number of sends allowed.
  int max_sends;
  /// Maximum number of receives allowed.
  int max_receives;
  /// Object chunk size, in bytes
  uint64_t object_chunk_size;
  /// The name of the plasma store socket to which plasma clients connect.
  std::string store_socket_name;
};

// TODO(hme): Add success/failure callbacks for push and pull.
class ObjectManager {
 public:

  using ObjectTransferCallback = std::function<void(const ObjectID &)>;

  /// Implicitly instantiates Ray implementation of ObjectDirectory.
  ///
  /// \param main_service The main asio io_service.
  /// \param config ObjectManager configuration.
  /// \param gcs_client A client connection to the Ray GCS.
  explicit ObjectManager(boost::asio::io_service &main_service,
                         const ObjectManagerConfig &config,
                         std::shared_ptr<gcs::AsyncGcsClient> gcs_client);

  /// Takes user-defined ObjectDirectoryInterface implementation.
  /// When this constructor is used, the ObjectManager assumes ownership of
  /// the given ObjectDirectory instance.
  ///
  /// \param main_service The main asio io_service.
  /// \param config ObjectManager configuration.
  /// \param od An object implementing the object directory interface.
  explicit ObjectManager(boost::asio::io_service &main_service,
                         const ObjectManagerConfig &config,
                         std::unique_ptr<ObjectDirectoryInterface> od);

  ~ObjectManager();

  /// Subscribe to notifications of objects added to local store.
  /// Upon subscribing, the callback will be invoked for all objects that
  ///
  /// already exist in the local store.
  /// \param callback The callback to invoke when objects are added to the local store.
  /// \return Status of whether adding the subscription succeeded.
  ray::Status SubscribeObjAdded(std::function<void(const ObjectInfoT &)> callback);

  /// Subscribe to notifications of objects deleted from local store.
  ///
  /// \param callback The callback to invoke when objects are removed from the local
  /// store.
  /// \return Status of whether adding the subscription succeeded.
  ray::Status SubscribeObjDeleted(std::function<void(const ray::ObjectID &)> callback);

  /// Push an object to to the node manager on the node corresponding to client id.
  ///
  /// \param object_id The object's object id.
  /// \param client_id The remote node's client id.
  /// \return Status of whether the push request successfully initiated.
  ray::Status Push(const ObjectID &object_id, const ClientID &client_id);

  /// Identify clients that contain the given object, and request
  /// the object from each one up to num_tries number of times.
  /// If the object does not appear locally after a certain period of time,
  /// invoke the failure callback. If the object appears locally, invoke the
  /// success callback. This method does nothing if an attempt to pull the given
  /// object is already in progress.
  ///
  /// \param object_id The object's object id.
  /// \return Status of whether the pull request successfully initiated. If an attempt
  /// to pull the given object is already in progress, this method will return status ok.
  ray::Status Pull(const ObjectID &object_id,
                   uint64_t num_attempts,
                   const ObjectTransferCallback &success,
                   const ObjectTransferCallback &failure);

  /// Discover ClientID via ObjectDirectory, then pull object
  /// from ClientID associated with ObjectID.
  ///
  /// \param object_id The object's object id.
  /// \param client_ids A vector of client_ids to try.
  /// \return Status of whether the pull request successfully initiated.
  ray::Status Pull(const ObjectID &object_id,
                   const ClientID &client_id,
                   uint64_t num_attempts,
                   const ObjectTransferCallback &success,
                   const ObjectTransferCallback &failure);

  /// Add a connection to a remote object manager.
  /// This is invoked by an external server.
  ///
  /// \param conn The connection.
  /// \return Status of whether the connection was successfully established.
  void ProcessNewClient(std::shared_ptr<TcpClientConnection> conn);

  /// Process messages sent from other nodes. We only establish
  /// transfer connections using this method; all other transfer communication
  /// is done separately.
  ///
  /// \param conn The connection.
  /// \param message_type The message type.
  /// \param message A pointer set to the beginning of the message.
  void ProcessClientMessage(std::shared_ptr<TcpClientConnection> conn,
                            int64_t message_type, const uint8_t *message);

  /// Cancels all requests (Push/Pull) associated with the given ObjectID.
  ///
  /// \param object_id The ObjectID.
  /// \return Status of whether requests were successfully cancelled.
  ray::Status Cancel(const ObjectID &object_id);

  /// Callback definition for wait.
  using WaitCallback = std::function<void(const ray::Status, uint64_t,
                                          const std::vector<ray::ObjectID> &)>;
  /// Wait for timeout_ms before invoking the provided callback.
  /// If num_ready_objects is satisfied before the timeout, then
  /// invoke the callback.
  ///
  /// \param object_ids The object ids to wait on.
  /// \param timeout_ms The time in milliseconds to wait before invoking the callback.
  /// \param num_ready_objects The minimum number of objects required before
  /// invoking the callback.
  /// \param callback Invoked when either timeout_ms is satisfied OR num_ready_objects
  /// is satisfied.
  /// \return Status of whether the wait successfully initiated.
  ray::Status Wait(const std::vector<ObjectID> &object_ids, uint64_t timeout_ms,
                   int num_ready_objects, const WaitCallback &callback);

 private:

  const uint64_t default_num_pull_attempts_ = 3;

  ClientID client_id_;
  const ObjectManagerConfig config_;
  std::unique_ptr<ObjectDirectoryInterface> object_directory_;
  ObjectStoreNotificationManager store_notification_;
  ObjectBufferPool buffer_pool_;

  /// This runs on a thread pool dedicated to sending objects.
  boost::asio::io_service send_service_;
  /// This runs on a thread pool dedicated to receiving objects.
  boost::asio::io_service receive_service_;

  /// Weak reference to main service. We ensure this object is destroyed before
  /// main_service_ is stopped.
  boost::asio::io_service *main_service_;

  /// Used to create "work" for send_service_.
  /// Without this, if send_service_ has no more sends to process, it will stop.
  boost::asio::io_service::work send_work_;
  /// Used to create "work" for receive_service_.
  /// Without this, if receive_service_ has no more receives to process, it will stop.
  boost::asio::io_service::work receive_work_;

  /// Runs the send service, which handle
  /// all outgoing object transfers.
  std::vector<std::thread> send_threads_;
  /// Runs the receive service, which handle
  /// all incoming object transfers.
  std::vector<std::thread> receive_threads_;

  /// Connection pool for reusing outgoing connections to remote object managers.
  ConnectionPool connection_pool_;

  /// Cache of locally available objects.
  std::unordered_map<ObjectID, ObjectInfoT, UniqueIDHasher> local_objects_;

  enum class ObjectTransferState : int {
    IDLE=0,
    PULL_REQUEST,
    PULL_RECEIVE,
    FAILED
  };

  struct ObjectTransferClientInfo {
    ObjectTransferClientInfo() = default;
    ObjectTransferClientInfo(uint64_t max_attemps,
                             const ObjectTransferCallback &success,
                             const ObjectTransferCallback &failure)
        : max_attempts(max_attempts), success(success), fail(failure){}
    ObjectTransferState state = ObjectTransferState::IDLE;
    uint64_t num_attempts = 0;
    uint64_t max_attempts;
    const ObjectTransferCallback &success;
    const ObjectTransferCallback &fail;
  };

  struct ObjectTransfer {
    ObjectTransfer() = default;
    ObjectTransfer(asio::io_service service,
                   uint64_t timout_ms,
                   uint64_t num_reties,
                   const ObjectTransferCallback &success,
                   const ObjectTransferCallback &failure) :
        timer(service, boost::posix_time::milliseconds(timout_ms)), default_num_attempts(num_reties), success(success), fail(failure){};
    boost::asio::deadline_timer timer;
    uint64_t client_cursor = 0;
    uint64_t num_get_location_attempts = 0;
    uint64_t default_num_attempts;
    std::vector<ClientID> client_ids;
    std::unordered_map<ClientID, ObjectTransferClientInfo, UniqueID> client_info;
    const ObjectTransferCallback &success;
    const ObjectTransferCallback &fail;
  };

  std::unordered_map<ObjectID, ObjectTransfer, UniqueID> object_transfers_;

  bool ObjectInTransit(const ObjectID &object_id){
    return object_transfers_.count(object_id) > 0;
  }

  bool ObjectLocal(const ObjectID &object_id){
    return local_objects_.count(object_id) > 0;
  }

  void CreateObjectTransfer(const ObjectID &object_id, uint64_t num_attempts,
                            asio::io_service service,
                            uint64_t timout_ms,
                            const ObjectTransferCallback &success,
                            const ObjectTransferCallback &failure){
    object_transfers_.emplace(object_id, std::move(ObjectTransfer(service,
                                                                  timout_ms,
        num_attempts,
        success,
        failure
    )));
  }

  /// This is the only method that can remove an object transfer.
  /// This is called when we receive notification from the object store that the object
  /// has been added.
  void RemoveObjectTransfer(const ObjectID &object_id){
    object_transfers_[object_id].timer.cancel();
    object_transfers_.erase(object_id);
  }

  uint64_t DefaultNumRetries(const ObjectID &object_id){
    return object_transfers_[object_id].default_num_attempts;
  }

  void AddObjectTransferClients(const ObjectID &object_id,
                                const std::vector<ClientID> &new_client_ids,
                                uint64_t num_attempts) {
    for (const ClientID &client_id : new_client_ids) {
      AddObjectTransferClient(object_id, client_id, num_attempts, nullptr, nullptr);
    }
  }

  void AddObjectTransferClient(const ObjectID &object_id,
                               const ClientID &client_id,
                               uint64_t num_attempts,
                               const ObjectTransferCallback &success,
                               const ObjectTransferCallback &failure){
    if (client_id == client_id_){
      // Don't pull from self.
      return;
    }
    ObjectTransfer &object_transfer = object_transfers_[object_id];
    if (object_transfer.client_info.count(client_id) == 0){
      object_transfer.client_ids.push_back(client_id);
      object_transfer.client_info.emplace(client_id, std::move(ObjectTransferClientInfo(num_attempts, success, failure)));
    } else {
      if (object_transfer.client_info[client_id].state == ObjectTransferState::FAILED){
        // Retry this client if it has been marked failed.
        object_transfer.client_info[client_id].state = ObjectTransferState::IDLE;
        object_transfer.client_info[client_id].num_attempts = 0;
        // Move it to the end of the vector.
        std::vector<ClientID> v = object_transfer.client_ids;
        v.erase(std::remove(v.begin(), v.end(), client_id), v.end());
        v.push_back(client_id);
      }
    }
  }

  /// This is invoked when an object is being received from a remote client.
  /// This is the only other method other than Pull that
  /// creates an object transfer.
  void AddIncomingTransfer(const ObjectID &object_id,
                           const ClientID &client_id){
    if(ObjectLocal(object_id)) {
      // Do nothing.
      return;
    }
    if (ObjectInTransit(object_id)){
      // Pull is already being processed, either from this method or a Pull invocation.
      AddObjectTransferClient(object_id, client_id, DefaultNumRetries(object_id),
                              nullptr,
                              nullptr);
    } else {
      // Object is not local and no pull request has been created.
      CreateObjectTransfer(object_id, default_num_pull_attempts_, *main_service_,
                           config_.pull_timeout_ms, nullptr, nullptr);
      AddObjectTransferClient(object_id, client_id, default_num_pull_attempts_, nullptr,
                              nullptr);

    }
    ObjectTransfer &object_transfer = object_transfers_[object_id];
    object_transfer.client_info[client_id].state = ObjectTransferState::PULL_RECEIVE;
  }

  const ClientID &GetNextClient(const ObjectID &object_id){
    // Iterate over every client round robin, and mark each one as failed only
    // after trying the client max_attempts number of times.
    // Return nil ClientID once all clients have been tried max_attempts number of times..
    ObjectTransfer &object_transfer = object_transfers_[object_id];
    int num_failed = 0;
    while(num_failed < object_transfer.client_ids.size()) {
      const ClientID
          &client_id = object_transfer.client_ids[object_transfer.client_cursor];
      ObjectTransferClientInfo &client_info = object_transfer.client_info[client_id];
      switch (client_info.state) {
        case ObjectTransferState::IDLE:
        // We can try this client.
        {
          client_info.state = ObjectTransferState::PULL_REQUEST;
          return client_id;
        }
        case ObjectTransferState::PULL_REQUEST:
        // A pull request for this client has already gone out.
        // Mark it as failed or idle, and cycle through remaining clients.
        case ObjectTransferState::PULL_RECEIVE:
        // A pull is currently being received from this client, and we've timed out.
        // Handle the same way we handle PULL_REQUEST.
        {
          client_info.num_attempts += 1;
          if (client_info.num_attempts >= client_info.max_attempts){
            client_info.state = ObjectTransferState::FAILED;
          } else {
            client_info.state = ObjectTransferState::IDLE;
          }
          object_transfer.client_cursor = (object_transfer.client_cursor + 1) % object_transfer.client_ids.size();
          break;
        }
        case ObjectTransferState::FAILED:
        // This is a failed transfer, so increment num_failed.
        {
          num_failed += 1;
        }
      }
    }
    // If all clients fail, return nil. The Pull attempt is a failure.
    return ClientID::nil();
  }

  /// Wait pull_timeout_ms milliseconds before attempting to obtain locations for
  /// object_id.
  /// This is invoked when GetLocationsFailed is invoked.
  void RetryGetLocations(const ObjectID &object_id){
    ObjectTransfer &object_transfer = object_transfers_[object_id];
    object_transfer.num_get_location_attempts += 1;
    if (object_transfer.num_get_location_attempts < object_transfer.default_num_attempts) {
      object_transfer.timer.async_wait(
          [this, object_id](const boost::system::error_code &error_code) {
            if (error_code.value() == 0) {
              RAY_CHECK_OK(PullGetLocations(object_id));
            }
          });
    } else {
      PullFailed(object_id);
    }
  }

  /// Wait config_.pull_timeout_ms milliseconds before trying to a pull object_id
  /// from another client, or retrying known clients.
  /// This is invoked if a pull fails at any point OR when a pull request is sent.
  void RetryPullAfterTimeout(const ObjectID &object_id){
    object_transfers_[object_id].timer.async_wait(
        [this, object_id](const boost::system::error_code &error_code) {
          if (error_code.value() == 0){
            TryPull(object_id);
          }
        });
  }

  void TryPull(const ObjectID &object_id){
    const ClientID &client_id = GetNextClient(object_id);
    if (client_id.is_nil()){
      // The pull attempt has failed.
      PullFailed(object_id);
    } else {
      // Establish a connection with the remote node in order to submit the pull request.
      PullEstablishConnection(object_id, client_id);
    }
  }

  void PullFailed(const ObjectID &object_id){
    if (object_transfers_[object_id].fail != nullptr) {
      object_transfers_[object_id].fail(object_id);
    }
    for (auto id_info : object_transfers_[object_id].client_info){
      const ObjectTransferClientInfo &client_info = id_info.second;
      if (client_info.fail != nullptr) {
        client_info.fail(object_id);
      }
    }
    // invoke any pull requests for
    RemoveObjectTransfer(object_id);
  }

  void PullSucceeded(const ObjectID &object_id){
    // Did we ever initiate a pull request for this object?
    if(object_transfers_.count(object_id) > 0){
      if (object_transfers_[object_id].success != nullptr){
        object_transfers_[object_id].success(object_id);
      }
      for (auto id_info : object_transfers_[object_id].client_info){
        const ObjectTransferClientInfo &client_info = id_info.second;
        if (client_info.success != nullptr) {
          client_info.success(object_id);
        }
      }
      RemoveObjectTransfer(object_id);
    }
  }

  /// Handle starting, running, and stopping asio io_service.
  void StartIOService();
  void RunSendService();
  void RunReceiveService();
  void StopIOService();

  /// Register object add with directory.
  void NotifyDirectoryObjectAdd(const ObjectInfoT &object_info);

  /// Register object remove with directory.
  void NotifyDirectoryObjectDeleted(const ObjectID &object_id);

  /// Part of an asynchronous sequence of Pull methods.
  /// Gets the location of an object before invoking PullEstablishConnection.
  /// Guaranteed to execute on main_service_ thread.
  /// Executes on main_service_ thread.
  ray::Status PullGetLocations(const ObjectID &object_id);

  /// Part of an asynchronous sequence of Pull methods.
  /// Uses an existing connection or creates a connection to ClientID.
  /// Executes on main_service_ thread.
  ray::Status PullEstablishConnection(const ObjectID &object_id,
                                      const ClientID &client_id);

  /// Private callback implementation for success on get location. Called from
  /// ObjectDirectory.
  void GetLocationsSuccess(const std::vector<ray::ClientID> &client_ids,
                           const ray::ObjectID &object_id);

  /// Private callback implementation for failure on get location. Called from
  /// ObjectDirectory.
  void GetLocationsFailed(const ObjectID &object_id);

  /// Synchronously send a pull request via remote object manager connection.
  /// Executes on main_service_ thread.
  ray::Status PullSendRequest(const ObjectID &object_id,
                              std::shared_ptr<SenderConnection> conn);

  std::shared_ptr<SenderConnection> CreateSenderConnection(
      ConnectionPool::ConnectionType type, RemoteConnectionInfo info);

  /// Begin executing a send.
  /// Executes on send_service_ thread pool.
  void ExecuteSendObject(const ClientID &client_id, const ObjectID &object_id,
                         uint64_t data_size, uint64_t metadata_size, uint64_t chunk_index,
                         const RemoteConnectionInfo &connection_info);
  /// This method synchronously sends the object id and object size
  /// to the remote object manager.
  /// Executes on send_service_ thread pool.
  ray::Status SendObjectHeaders(const ObjectID &object_id, uint64_t data_size,
                                uint64_t metadata_size, uint64_t chunk_index,
                                std::shared_ptr<SenderConnection> conn);

  /// This method initiates the actual object transfer.
  /// Executes on send_service_ thread pool.
  ray::Status SendObjectData(const ObjectID &object_id,
                             const ObjectBufferPool::ChunkInfo &chunk_info,
                             std::shared_ptr<SenderConnection> conn);

  /// Invoked when a remote object manager pushes an object to this object manager.
  /// This will invoke the object receive on the receive_service_ thread pool.
  void ReceivePushRequest(std::shared_ptr<TcpClientConnection> conn,
                          const uint8_t *message);
  /// Execute a receive on the receive_service_ thread pool.
  void ExecuteReceiveObject(const ClientID &client_id, const ObjectID &object_id,
                            uint64_t data_size, uint64_t metadata_size,
                            uint64_t chunk_index,
                            std::shared_ptr<TcpClientConnection> conn);

  /// Handles receiving a pull request message.
  void ReceivePullRequest(std::shared_ptr<TcpClientConnection> &conn,
                          const uint8_t *message);

  /// Handles connect message of a new client connection.
  void ConnectClient(std::shared_ptr<TcpClientConnection> &conn, const uint8_t *message);
  /// Handles disconnect message of an existing client connection.
  void DisconnectClient(std::shared_ptr<TcpClientConnection> &conn,
                        const uint8_t *message);
};

}  // namespace ray

#endif  // RAY_OBJECT_MANAGER_OBJECT_MANAGER_H
