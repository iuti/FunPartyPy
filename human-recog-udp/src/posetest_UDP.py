import socket
import pickle

class UDPReceiver:
    def __init__(self, ip='0.0.0.0', port=9999):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))
        print(f"Listening for UDP packets on {self.ip}:{self.port}")

    def receive_data(self):
        while True:
            # Receive the size of the incoming data
            size_data, addr = self.sock.recvfrom(4)
            size = int.from_bytes(size_data, byteorder='big')

            # Receive the actual data
            data, addr = self.sock.recvfrom(size)
            image = pickle.loads(data)

            # Process the received image (for demonstration, we just print the size)
            print(f"Received image of size: {len(data)} bytes from {addr}")

def main():
    receiver = UDPReceiver()
    receiver.receive_data()

if __name__ == "__main__":
    main()