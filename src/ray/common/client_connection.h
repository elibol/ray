#ifndef CLIENT_CONNECTION_H
#define CLIENT_CONNECTION_H

#include <boost/asio.hpp>
#include <boost/asio/error.hpp>
#include <boost/enable_shared_from_this.hpp>
#include <memory>

namespace ray {

template <class T>
class ClientManager;

/// \class ClientConnection
///
/// A generic type representing a client connection on a server. This class can
/// be used to process and write messages asynchronously from and to the
/// client.
template <class T>
class ClientConnection : public std::enable_shared_from_this<ClientConnection<T>> {

 public:

  typedef std::shared_ptr<ClientConnection<T>> pointer;

  /// Allocate a new node client connection.
  ///
  /// \param ClientManager A reference to the manager that will process a
  /// message from this client.
  /// \param socket The client socket.
  /// \return std::shared_ptr<ClientConnection>.
  static pointer Create(ClientManager<T>& manager,
                        boost::asio::basic_stream_socket<T> &&socket);

  /// Allocate a new node client connection.
  /// \param ClientManager A reference to the manager that will process a
  /// \param io_service The asio io_service.
  /// message from this client.
  static pointer Create(ClientManager<T>& manager,
                        boost::asio::io_service& io_service);

  /// Listen for and process messages from the client connection. Once a
  /// message has been fully received, the client manager's
  /// ProcessClientMessage handler will be called.
  void ProcessMessages();

  /// Write a message to the client and then listen for more messages.
  ///
  /// \param type The message type (e.g., a flatbuffer enum).
  /// \param length The size in bytes of the message.
  /// \param message A pointer to the message buffer. This will be copied into
  /// the ClientConnection's buffer.
  void WriteMessage(int64_t type, size_t length, const uint8_t *message);

  /// \return The socket associated with this connection.
  boost::asio::basic_stream_socket<T> &GetSocket();

 private:
  /// A private constructor for a node client connection with socket.
  ClientConnection(
      ClientManager<T>& manager,
      boost::asio::basic_stream_socket<T> &&socket);
  /// A private constructor for a node client connection.
  ClientConnection(ClientManager<T>& manager, boost::asio::io_service& io_service);
  /// Process an error from the last operation, then process the  message
  /// header from the client.
  void processMessageHeader(const boost::system::error_code& error);
  /// Process an error from reading the message header, then process the
  /// message from the client.
  void processMessage(const boost::system::error_code& error);
  /// Process an error from the last operation and then listen for more
  /// messages.
  void processMessages(const boost::system::error_code& error);

  /// The client socket.
  boost::asio::basic_stream_socket<T> socket_;
  /// A reference to the manager for this client. The manager exposes a handler
  /// for all messages processed by this client.
  ClientManager<T>& manager_;
  /// Buffers for the current message being read from the client.
  int64_t read_version_;
  int64_t read_type_;
  uint64_t read_length_;
  std::vector<uint8_t> read_message_;
  /// Buffers for the current message being written to the client.
  int64_t write_version_;
  int64_t write_type_;
  uint64_t write_length_;
  std::vector<uint8_t> write_message_;

};

using LocalClientConnection = ClientConnection<boost::asio::local::stream_protocol>;
using TcpClientConnection = ClientConnection<boost::asio::ip::tcp>;

/// \class ClientManager
///
/// A virtual client manager. Derived classes should define a method for
/// processing a message on the server sent by the client.
template <class T>
class ClientManager {
 public:
  /// Process a new client connection.
  ///
  /// \param client A shared pointer to the client that connected.
  virtual void ProcessNewClient(std::shared_ptr<ClientConnection<T>> client) =
    0;

  /// Process a message from a client, then listen for more messages if the
  /// client is still alive.
  ///
  /// \param client A shared pointer to the client that sent the message.
  /// \param message_type The message type (e.g., a flatbuffer enum).
  /// \param message A pointer to the message buffer.
  virtual void ProcessClientMessage(
      std::shared_ptr<ClientConnection<T>> client,
      int64_t message_type,
      const uint8_t *message) = 0;

  virtual ~ClientManager() = 0;
};

} // end namespace ray

#endif
