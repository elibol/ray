#include "ray/object_manager/object_manager.h"

namespace asio = boost::asio;

namespace ray {

ObjectManager::ObjectManager(asio::io_service &main_service,
                             std::unique_ptr<asio::io_service> object_manager_service,
                             const ObjectManagerConfig &config,
                             std::shared_ptr<gcs::AsyncGcsClient> gcs_client)
    // TODO(hme): Eliminate knowledge of GCS.
    : client_id_(gcs_client->client_table().GetLocalClientId()),
      object_directory_(new ObjectDirectory(gcs_client)),
      store_notification_(main_service, config.store_socket_name),
      store_pool_(config.store_socket_name),
      object_manager_service_(std::move(object_manager_service)),
      work_(*object_manager_service_),
      connection_pool_(object_directory_.get(), &*object_manager_service_, client_id_),
      transfer_queue_(),
      num_transfers_send_(0),
      num_transfers_receive_(0) {
  main_service_ = &main_service;
  config_ = config;
  store_notification_.SubscribeObjAdded(
      [this](const ObjectID &oid) { NotifyDirectoryObjectAdd(oid); });
  store_notification_.SubscribeObjDeleted(
      [this](const ObjectID &oid) { NotifyDirectoryObjectDeleted(oid); });
  StartIOService();
}

ObjectManager::ObjectManager(asio::io_service &main_service,
                             std::unique_ptr<asio::io_service> object_manager_service,
                             const ObjectManagerConfig &config,
                             std::unique_ptr<ObjectDirectoryInterface> od)
    : object_directory_(std::move(od)),
      store_notification_(main_service, config.store_socket_name),
      store_pool_(config.store_socket_name),
      object_manager_service_(std::move(object_manager_service)),
      work_(*object_manager_service_),
      connection_pool_(object_directory_.get(), &*object_manager_service_, client_id_),
      transfer_queue_(),
      num_transfers_send_(0),
      num_transfers_receive_(0) {
  // TODO(hme) Client ID is never set with this constructor.
  main_service_ = &main_service;
  config_ = config;
  store_notification_.SubscribeObjAdded(
      [this](const ObjectID &oid) { NotifyDirectoryObjectAdd(oid); });
  store_notification_.SubscribeObjDeleted(
      [this](const ObjectID &oid) { NotifyDirectoryObjectDeleted(oid); });
  StartIOService();
}

void ObjectManager::StartIOService() {
  for (int i = 0; i < config_.num_threads; ++i) {
    io_threads_.emplace_back(std::thread(&ObjectManager::IOServiceLoop, this));
  }
}

void ObjectManager::IOServiceLoop() { object_manager_service_->run(); }

void ObjectManager::StopIOService() {
  object_manager_service_->stop();
  for (int i = 0; i < config_.num_threads; ++i) {
    io_threads_[i].join();
  }
}

void ObjectManager::NotifyDirectoryObjectAdd(const ObjectID &object_id) {
  local_objects_.insert(object_id);
  ray::Status status = object_directory_->ReportObjectAdded(object_id, client_id_);
}

void ObjectManager::NotifyDirectoryObjectDeleted(const ObjectID &object_id) {
  local_objects_.erase(object_id);
  ray::Status status = object_directory_->ReportObjectRemoved(object_id, client_id_);
}

ray::Status ObjectManager::Terminate() {
  StopIOService();
  ray::Status status_code = object_directory_->Terminate();
  // TODO: evaluate store client termination status.
  store_notification_.Terminate();
  store_pool_.Terminate();
  return status_code;
}

ray::Status ObjectManager::SubscribeObjAdded(
    std::function<void(const ObjectID &)> callback) {
  store_notification_.SubscribeObjAdded(callback);
  return ray::Status::OK();
}

ray::Status ObjectManager::SubscribeObjDeleted(
    std::function<void(const ObjectID &)> callback) {
  store_notification_.SubscribeObjDeleted(callback);
  return ray::Status::OK();
}

ray::Status ObjectManager::Pull(const ObjectID &object_id) {
  main_service_->dispatch(
      [this, object_id]() { RAY_CHECK_OK(PullGetLocations(object_id)); });
  return Status::OK();
}

void ObjectManager::SchedulePull(const ObjectID &object_id, int wait_ms) {
  pull_requests_[object_id] = std::shared_ptr<boost::asio::deadline_timer>(
      new asio::deadline_timer(*main_service_, boost::posix_time::milliseconds(wait_ms)));
  pull_requests_[object_id]->async_wait(
      [this, object_id](const boost::system::error_code &error_code) {
        pull_requests_.erase(object_id);
        main_service_->dispatch(
            [this, object_id]() { RAY_CHECK_OK(PullGetLocations(object_id)); });
      });
}

ray::Status ObjectManager::PullGetLocations(const ObjectID &object_id) {
  ray::Status status_code = object_directory_->GetLocations(
      object_id,
      [this](const std::vector<ClientID> &client_ids, const ObjectID &object_id) {
        return GetLocationsSuccess(client_ids, object_id);
      },
      [this](const ObjectID &object_id) { return GetLocationsFailed(object_id); });
  return status_code;
}

void ObjectManager::GetLocationsSuccess(const std::vector<ray::ClientID> &client_ids,
                                        const ray::ObjectID &object_id) {
  RAY_CHECK(!client_ids.empty());
  ClientID client_id = client_ids.front();
  pull_requests_.erase(object_id);
  ray::Status status_code = Pull(object_id, client_id);
}

void ObjectManager::GetLocationsFailed(const ObjectID &object_id) {
  SchedulePull(object_id, config_.pull_timeout_ms);
}

ray::Status ObjectManager::Pull(const ObjectID &object_id, const ClientID &client_id) {
  main_service_->dispatch([this, object_id, client_id]() {
    RAY_CHECK_OK(PullEstablishConnection(object_id, client_id));
  });
  return Status::OK();
};

ray::Status ObjectManager::PullEstablishConnection(const ObjectID &object_id,
                                                   const ClientID &client_id) {
  // Check if object is already local, and client_id is not itself.
  if (local_objects_.count(object_id) != 0 || client_id == client_id_) {
    return ray::Status::OK();
  }

  Status status = connection_pool_.GetSender(
      ConnectionPool::ConnectionType::MESSAGE, client_id,
      [this, object_id](boost::shared_ptr<SenderConnection> conn) {
        Status status = PullSendRequest(object_id, conn);
      },
      [this, object_id]() {
        // connection failed, so reschedule pull.
        SchedulePull(object_id, config_.pull_timeout_ms);
      });
  return status;
}

ray::Status ObjectManager::PullSendRequest(const ObjectID &object_id,
                                           boost::shared_ptr<SenderConnection> conn) {
  flatbuffers::FlatBufferBuilder fbb;
  auto message = CreatePullRequestMessage(fbb, fbb.CreateString(client_id_.binary()),
                                          fbb.CreateString(object_id.binary()));
  fbb.Finish(message);
  RAY_CHECK_OK(conn->WriteMessage(OMMessageType_PullRequest, fbb.GetSize(),
                                  fbb.GetBufferPointer()));
  RAY_CHECK_OK(
      connection_pool_.ReleaseSender(ConnectionPool::ConnectionType::MESSAGE, conn));
  return ray::Status::OK();
}

ray::Status ObjectManager::Push(const ObjectID &object_id, const ClientID &client_id) {
  transfer_queue_.QueueSend(client_id, object_id);
  return DequeueTransfers();
}

ray::Status ObjectManager::DequeueTransfers() {
  ray::Status status = ray::Status::OK();
  // Dequeue sends.
  while (true) {
    if(std::atomic_fetch_add(&num_transfers_send_, 1) <= config_.max_sends) {
      TransferQueue::SendRequest req;
      bool exists = transfer_queue_.DequeueSendIfPresent(&req);
      if (exists) {
        object_manager_service_->dispatch([this, req]() {
          RAY_LOG(DEBUG) << "DequeueSend " << client_id_ << " " << req.object_id << " "
                         << num_transfers_send_ << "/" << config_.max_sends;
          RAY_CHECK_OK(ExecuteSendObject(req.object_id, req.client_id));
        });
      } else {
        std::atomic_fetch_sub(&num_transfers_send_, 1);
        break;
      }
    } else {
      std::atomic_fetch_sub(&num_transfers_send_, 1);
      break;
    }
  }
  // Dequeue receives.
  while (true) {
    if(std::atomic_fetch_add(&num_transfers_receive_, 1) <= config_.max_receives) {
      TransferQueue::ReceiveRequest req;
      bool exists = transfer_queue_.DequeueReceiveIfPresent(&req);
      if (exists) {
        object_manager_service_->dispatch([this, req]() {
          RAY_LOG(DEBUG) << "DequeueReceive " << client_id_ << " " << req.object_id << " "
                         << num_transfers_receive_ << "/" << config_.max_receives;
          RAY_CHECK_OK(ExecuteReceiveObject(req.client_id, req.object_id, req.object_size,
                                            req.conn));
        });
      } else {
        std::atomic_fetch_sub(&num_transfers_receive_, 1);
        break;
      }
    } else {
      std::atomic_fetch_sub(&num_transfers_receive_, 1);
      break;
    }
  }
  return status;
}

ray::Status ObjectManager::TransferCompleted(TransferQueue::TransferType type) {
  if (type == TransferQueue::TransferType::SEND) {
    std::atomic_fetch_sub(&num_transfers_send_, 1);
  } else {
    std::atomic_fetch_sub(&num_transfers_receive_, 1);
  }
  return DequeueTransfers();
};

ray::Status ObjectManager::ExecuteSendObject(const ObjectID &object_id,
                                             const ClientID &client_id) {
  ray::Status status;
  status = connection_pool_.GetSender(
      ConnectionPool::ConnectionType::TRANSFER, client_id,
      [this, object_id](boost::shared_ptr<SenderConnection> conn) {
        ray::Status status = SendObjectHeaders(object_id, conn);
        if (!status.ok()) {
          // TODO(hme): Keep track of retries,
          // and only retry on object not local
          // for now.
          Push(object_id, conn->GetClientID());
        }
      },
      [this, object_id]() {
        // Push is best effort, so do nothing on failure.
      });
  return status;
}

ray::Status ObjectManager::SendObjectHeaders(const ObjectID &object_id_const,
                                             boost::shared_ptr<SenderConnection> conn) {
  ObjectID object_id = ObjectID(object_id_const);
  // Allocate and append the request to the transfer queue.
  plasma::ObjectBuffer object_buffer;
  plasma::ObjectID plasma_id = object_id.to_plasma_id();
  std::shared_ptr<plasma::PlasmaClient> store_client = store_pool_.GetObjectStore();
  ARROW_CHECK_OK(store_client->Get(&plasma_id, 1, 0, &object_buffer));
  if (object_buffer.data_size == -1) {
    RAY_LOG(ERROR) << "Failed to get object";
    // If the object wasn't locally available, exit immediately. If the object
    // later appears locally, the requesting plasma manager should request the
    // transfer again.
    RAY_CHECK_OK(
        connection_pool_.ReleaseSender(ConnectionPool::ConnectionType::TRANSFER, conn));
    return ray::Status::IOError(
        "Unable to transfer object to requesting plasma manager, object not local.");
  }
  RAY_CHECK(object_buffer.metadata->data() ==
            object_buffer.data->data() + object_buffer.data_size);

  TransferQueue::SendContext context;
  context.client_id = conn->GetClientID();
  context.object_id = object_id;
  context.object_size = static_cast<uint64_t>(object_buffer.data_size);
  context.data = const_cast<uint8_t *>(object_buffer.data->data());
  UniqueID context_id = transfer_queue_.AddContext(context);

  // Create buffer.
  flatbuffers::FlatBufferBuilder fbb;
  // TODO(hme): use to_flatbuf
  auto message = CreatePushRequestMessage(fbb, fbb.CreateString(object_id.binary()),
                                          context.object_size);
  fbb.Finish(message);
  ray::Status status = conn->WriteMessage(OMMessageType_PushRequest, fbb.GetSize(),
                                          fbb.GetBufferPointer());
  if (!status.ok()) {
    // push failed.
    // TODO(hme): Trash sender.
    RAY_CHECK_OK(
        connection_pool_.ReleaseSender(ConnectionPool::ConnectionType::TRANSFER, conn));
    return status;
  }

  // TODO(hme): Make this async again.
  return SendObjectData(conn, context_id, store_client);
}

ray::Status ObjectManager::SendObjectData(
    boost::shared_ptr<SenderConnection> conn, const UniqueID &context_id,
    std::shared_ptr<plasma::PlasmaClient> store_client) {
  TransferQueue::SendContext context = transfer_queue_.GetContext(context_id);
  boost::system::error_code ec;
  std::vector<asio::const_buffer> buffer;
  buffer.push_back(asio::buffer(context.data, context.object_size));
  conn->WriteBuffer(buffer, ec);

  ray::Status status = ray::Status::OK();
  if (ec.value() != 0) {
    // push failed.
    // TODO(hme): Trash sender.
    status = ray::Status::IOError(ec.message());
  }

  // Do this regardless of whether it failed or succeeded.
  ARROW_CHECK_OK(store_client->Release(context.object_id.to_plasma_id()));
  store_pool_.ReleaseObjectStore(store_client);
  RAY_CHECK_OK(
      connection_pool_.ReleaseSender(ConnectionPool::ConnectionType::TRANSFER, conn));
  RAY_CHECK_OK(transfer_queue_.RemoveContext(context_id));
  RAY_LOG(DEBUG) << "SendCompleted " << client_id_ << " " << context.object_id << " "
                 << num_transfers_send_ << "/" << config_.max_sends;
  RAY_CHECK_OK(TransferCompleted(TransferQueue::TransferType::SEND));
  return status;
}

ray::Status ObjectManager::Cancel(const ObjectID &object_id) {
  // TODO(hme): Account for pull timers.
  ray::Status status = object_directory_->Cancel(object_id);
  return ray::Status::OK();
}

ray::Status ObjectManager::Wait(const std::vector<ObjectID> &object_ids,
                                uint64_t timeout_ms, int num_ready_objects,
                                const WaitCallback &callback) {
  // TODO: Implement wait.
  return ray::Status::OK();
}

void ObjectManager::ProcessNewClient(std::shared_ptr<ReceiverConnection> conn) {
  conn->ProcessMessages();
}

void ObjectManager::ProcessClientMessage(std::shared_ptr<ReceiverConnection> conn,
                                         int64_t message_type, const uint8_t *message) {
  switch (message_type) {
  case OMMessageType_PushRequest: {
    ReceivePushRequest(conn, message);
    break;
  }
  case OMMessageType_PullRequest: {
    ReceivePullRequest(conn, message);
    break;
  }
  case OMMessageType_ConnectClient: {
    ConnectClient(conn, message);
    break;
  }
  case OMMessageType_DisconnectClient: {
    DisconnectClient(conn, message);
    break;
  }
  default: { RAY_LOG(FATAL) << "invalid request " << message_type; }
  }
}

void ObjectManager::ConnectClient(std::shared_ptr<ReceiverConnection> &conn,
                                  const uint8_t *message) {
  // TODO: trash connection on failure.
  auto info = flatbuffers::GetRoot<ConnectClientMessage>(message);
  ClientID client_id = ObjectID::from_binary(info->client_id()->str());
  bool is_transfer = info->is_transfer();
  conn->SetClientID(client_id);
  if (is_transfer) {
    connection_pool_.RegisterReceiver(ConnectionPool::ConnectionType::TRANSFER, client_id,
                                      conn);
  } else {
    connection_pool_.RegisterReceiver(ConnectionPool::ConnectionType::MESSAGE, client_id,
                                      conn);
  }
  conn->ProcessMessages();
}

void ObjectManager::DisconnectClient(std::shared_ptr<ReceiverConnection> &conn,
                                     const uint8_t *message) {
  auto info = flatbuffers::GetRoot<DisconnectClientMessage>(message);
  ClientID client_id = ObjectID::from_binary(info->client_id()->str());
  bool is_transfer = info->is_transfer();
  if (is_transfer) {
    connection_pool_.RemoveReceiver(ConnectionPool::ConnectionType::TRANSFER, client_id,
                                    conn);
  } else {
    connection_pool_.RemoveReceiver(ConnectionPool::ConnectionType::MESSAGE, client_id,
                                    conn);
  }
}

void ObjectManager::ReceivePullRequest(std::shared_ptr<ReceiverConnection> &conn,
                                       const uint8_t *message) {
  // Serialize and push object to requesting client.
  auto pr = flatbuffers::GetRoot<PullRequestMessage>(message);
  ObjectID object_id = ObjectID::from_binary(pr->object_id()->str());
  ClientID client_id = ClientID::from_binary(pr->client_id()->str());
  ray::Status push_status = Push(object_id, client_id);
  conn->ProcessMessages();
}

void ObjectManager::ReceivePushRequest(std::shared_ptr<ReceiverConnection> conn,
                                       const uint8_t *message) {
  // Serialize.
  auto object_header = flatbuffers::GetRoot<PushRequestMessage>(message);
  ObjectID object_id = ObjectID::from_binary(object_header->object_id()->str());
  int64_t object_size = (int64_t)object_header->object_size();
  transfer_queue_.QueueReceive(conn->GetClientID(), object_id, object_size, conn);
  RAY_CHECK_OK(DequeueTransfers());
}

ray::Status ObjectManager::ExecuteReceiveObject(
    ClientID client_id, ObjectID object_id, uint64_t object_size,
    std::shared_ptr<ReceiverConnection> conn) {
  boost::system::error_code ec;
  int64_t metadata_size = 0;
  // Try to create shared buffer.
  std::shared_ptr<Buffer> data;
  std::shared_ptr<plasma::PlasmaClient> store_client = store_pool_.GetObjectStore();
  arrow::Status s = store_client->Create(object_id.to_plasma_id(), object_size, NULL,
                                         metadata_size, &data);
  std::vector<boost::asio::mutable_buffer> buffer;
  if (s.ok()) {
    // Read object into store.
    uint8_t *mutable_data = data->mutable_data();
    buffer.push_back(asio::buffer(mutable_data, object_size));
    conn->ReadBuffer(buffer, ec);
    if (!ec.value()) {
      ARROW_CHECK_OK(store_client->Seal(object_id.to_plasma_id()));
      ARROW_CHECK_OK(store_client->Release(object_id.to_plasma_id()));
    } else {
      ARROW_CHECK_OK(store_client->Release(object_id.to_plasma_id()));
      ARROW_CHECK_OK(store_client->Abort(object_id.to_plasma_id()));
      RAY_LOG(ERROR) << "Receive Failed";
    }
  } else {
    RAY_LOG(ERROR) << "Buffer Create Failed: " << s.message();
    // Read object into empty buffer.
    uint8_t *mutable_data = (uint8_t *)malloc(object_size + metadata_size);
    buffer.push_back(asio::buffer(mutable_data, object_size));
    conn->ReadBuffer(buffer, ec);
  }
  store_pool_.ReleaseObjectStore(store_client);
  conn->ProcessMessages();
  RAY_LOG(DEBUG) << "ReceiveCompleted " << client_id_ << " " << object_id << " "
                 << num_transfers_receive_ << "/" << config_.max_receives;
  RAY_CHECK_OK(TransferCompleted(TransferQueue::TransferType::RECEIVE));
  return Status::OK();
}

}  // namespace ray
