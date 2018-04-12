#ifndef RAY_COMMON_CLIENT_CONNECTION_H
#define RAY_COMMON_CLIENT_CONNECTION_H

#include <memory>

#include <boost/asio.hpp>
#include <boost/asio/error.hpp>
#include <boost/enable_shared_from_this.hpp>

#include "ray/id.h"
#include "ray/status.h"

namespace ray {

/// Connect a TCP socket.
///
/// \param socket The socket to connect.
/// \param ip_address The IP address to connect to.
/// \param port The port to connect to.
/// \return Status.
ray::Status TcpConnect(boost::asio::ip::tcp::socket &socket,
                       const std::string &ip_address, int port);

/// \typename ServerConnection
///
/// A generic type representing a client connection to a server. This typename
/// can be used to write messages synchronously to the server.
template <typename T>
class ServerConnection {
 public:
  /// Create a connection to the server.
  ServerConnection(boost::asio::basic_stream_socket<T> &&socket);

  /// Write a message to the client.
  ///
  /// \param type The message type (e.g., a flatbuffer enum).
  /// \param length The size in bytes of the message.
  /// \param message A pointer to the message buffer.
  /// \return Status.
  ray::Status WriteMessage(int64_t type, int64_t length, const uint8_t *message);

  /// Write a buffer to this connection.
  ///
  /// \param buffer The buffer.
  /// \param ec The error code object in which to store error codes.
  void WriteBuffer(const std::vector<boost::asio::const_buffer> &buffer,
                   boost::system::error_code &ec);

  /// Read a buffer from this connection.
  ///
  /// \param buffer The buffer.
  /// \param ec The error code object in which to store error codes.
  void ReadBuffer(const std::vector<boost::asio::mutable_buffer> &buffer,
                  boost::system::error_code &ec);

  ray::Status AsyncWriteMessage(
      int64_t type, int64_t length, const uint8_t *message,
      std::function<void(const boost::system::error_code &error)> handler);

  /// Write a buffer to this connection asynchronously.
  ///
  /// \param buffer The buffer.
  /// \param handler Handler to invoke when write is completed.
  void AsyncWriteBuffer(
      const std::vector<boost::asio::const_buffer> &buffer,
      std::function<void(const boost::system::error_code &error)> handler);

  /// Read a buffer from this connection asynchronously.
  ///
  /// \param buffer The buffer.
  /// \param handler Handler to invoke when read is completed.
  void AsyncReadBuffer(
      const std::vector<boost::asio::mutable_buffer> &buffer,
      std::function<void(const boost::system::error_code &error)> handler);

  ray::Status Close(){
    boost::system::error_code ec;
    // This interrupts in-transit data (see documentation on close).
    socket_.close(ec);
    if (ec.value() != 0){
      return ray::Status::IOError(ec.message());
    }
    return ray::Status::OK();
  }

 protected:
  /// The socket connection to the server.
  boost::asio::basic_stream_socket<T> socket_;

 private:
  /// Handler for async read/write methods. This just calls the passed in handler.
  /// \param error The error code from writing the message.
  void AsyncCallHandler(
      std::function<void(const boost::system::error_code &error)> handler,
      const boost::system::error_code &error);
};

template <typename T>
class ClientConnection;

template <typename T>
using ClientHandler = std::function<void(std::shared_ptr<ClientConnection<T>>)>;
template <typename T>
using MessageHandler =
    std::function<void(std::shared_ptr<ClientConnection<T>>, int64_t, const uint8_t *)>;

/// \typename ClientConnection
///
/// A generic type representing a client connection on a server. In addition to
/// writing messages to the client, like in ServerConnection, this typename can
/// also be used to process messages asynchronously from client.
template <typename T>
class ClientConnection : public ServerConnection<T>,
                         public std::enable_shared_from_this<ClientConnection<T>> {
 public:
  /// Allocate a new node client connection.
  ///
  /// \param ClientManager A reference to the manager that will process a
  /// message from this client.
  /// \param socket The client socket.
  /// \return std::shared_ptr<ClientConnection>.
  static std::shared_ptr<ClientConnection<T>> Create(
      ClientHandler<T> &new_client_handler, MessageHandler<T> &message_handler,
      boost::asio::basic_stream_socket<T> &&socket);

  /// \return The ClientID of the remote client.
  const ClientID &GetClientID();

  /// \param client_id The ClientID of the remote client.
  void SetClientID(const ClientID &client_id);

  /// Listen for and process messages from the client connection. Once a
  /// message has been fully received, the client manager's
  /// ProcessClientMessage handler will be called.
  void ProcessMessages();

 private:
  /// A private constructor for a node client connection.
  ClientConnection(MessageHandler<T> &message_handler,
                   boost::asio::basic_stream_socket<T> &&socket);
  /// Process an error from the last operation, then process the  message
  /// header from the client.
  void ProcessMessageHeader(const boost::system::error_code &error);
  /// Process an error from reading the message header, then process the
  /// message from the client.
  void ProcessMessage(const boost::system::error_code &error);

  /// The ClientID of the remote client.
  ClientID client_id_;
  /// The handler for a message from the client.
  MessageHandler<T> message_handler_;
  /// Buffers for the current message being read rom the client.
  int64_t read_version_;
  int64_t read_type_;
  uint64_t read_length_;
  std::vector<uint8_t> read_message_;
};

using LocalServerConnection = ServerConnection<boost::asio::local::stream_protocol>;
using TcpServerConnection = ServerConnection<boost::asio::ip::tcp>;
using LocalClientConnection = ClientConnection<boost::asio::local::stream_protocol>;
using TcpClientConnection = ClientConnection<boost::asio::ip::tcp>;

}  // namespace ray

#endif  // RAY_COMMON_CLIENT_CONNECTION_H
