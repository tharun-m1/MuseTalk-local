setup_nginx() {
  echo "Setting up nginx..."
  sudo cp -rf ./default /etc/nginx/sites-available/default
  sudo service nginx restart
  sudo nginx -t
}
setup_nginx