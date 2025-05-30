sudo ufw enable
sudo ufw allow 80/tcp
sudo ufw allow 3478/udp    # Allow STUN
sudo ufw allow 3478/udp    # Allow TURN UDP
sudo ufw allow 443/tcp     # Allow TURN TCP (fallback)
sudo ufw allow 10000:65000/udp  # Allow WebRTC media
sudo ufw allow 443/tcp     # Allow HTTPS (signaling)
sudo ufw reload
sudo ufw status