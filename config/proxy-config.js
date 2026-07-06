// Proxy configurations
require('dotenv').config();

// PROXY_HOST tanimli degilse proxy'siz (dogrudan) baglanti icin null doner.
function getProxyConfig() {
  if (!process.env.PROXY_HOST) {
    return null;
  }

  return {
    server: `${process.env.PROXY_HOST}:${process.env.PROXY_PORT}`,
    username: process.env.PROXY_USERNAME || undefined,
    password: process.env.PROXY_PASSWORD || undefined,
  };
}

module.exports = { getProxyConfig };
