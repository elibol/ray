#ifndef RAYLET_CC
#define RAYLET_CC
#include "raylet.h"

#include <iostream>
#include <boost/bind.hpp>

#include "ray/status.h"

using namespace std;
using namespace boost::asio;

namespace ray {

Raylet::Raylet(boost::asio::io_service& io_service_nm,
               boost::asio::io_service& io_service_om,
               const std::string &socket_name,
               const ResourceSet &resource_config,
               const OMConfig &om_config,
               shared_ptr<ray::GcsClient> gcs_client)
    : acceptor_(io_service_nm, boost::asio::local::stream_protocol::endpoint(socket_name)),
      socket_(io_service_nm),
      tcp_acceptor_(io_service_om, ip::tcp::endpoint(ip::tcp::v4(), 0)),
      tcp_socket_(io_service_om),
      object_manager_(io_service_om, om_config, gcs_client),
      node_manager_(socket_name, resource_config, object_manager_),
      gcs_client_(gcs_client) {
  ClientID client_id = RegisterGcs();
  object_manager_.SetClientID(client_id);
  // Start listening for clients.
  DoAccept();
  DoAcceptTcp();
}

Raylet::~Raylet(){
  RAY_CHECK_OK(object_manager_.Terminate());
}

ClientID Raylet::RegisterGcs(){
  ip::tcp::endpoint endpoint = tcp_acceptor_.local_endpoint();
  std::string ip = endpoint.address().to_string();
  ushort port = endpoint.port();
  ClientID client_id = gcs_client_->Register(ip, port);
  return client_id;
}

void Raylet::DoAcceptTcp() {
  tcp_acceptor_.async_accept(
      // There is some asio magic that allows for the same socket to be used
      // repeatedly.
      tcp_socket_,
      boost::bind(&Raylet::HandleAcceptTcp, this, boost::asio::placeholders::error)
  );
}

void Raylet::HandleAcceptTcp(const boost::system::error_code& error) {
  if (!error) {
    auto new_connection = TcpClientConnection::Create(object_manager_, std::move(tcp_socket_));
    object_manager_.ProcessNewClient(new_connection);
  }
  DoAcceptTcp();
}

void Raylet::DoAccept() {
  acceptor_.async_accept(socket_,
      boost::bind(&Raylet::HandleAccept, this, boost::asio::placeholders::error)
      );
}

void Raylet::HandleAccept(const boost::system::error_code &error) {
  if (!error) {
    // Accept a new local client and dispatch it to the node manager.
    auto new_connection = LocalClientConnection::Create(node_manager_, std::move(socket_));
    node_manager_.ProcessNewClient(new_connection);
  }
  // We're ready to accept another client.
  DoAccept();
}

ObjectManager &Raylet::GetObjectManager() {
  return object_manager_;
}

} // namespace ray

#endif // RAYLET_CC
