// Proxy configurations
require('dotenv').config();

// Webshare'in disa aktardigi "host:port:username:password" formatindaki tek satiri proxy objesine cevirir.
function parseProxyEntry(entry) {
  const [host, port, username, password] = entry.split(':');
  return {
    server: `${host}:${port}`,
    username: username || undefined,
    password: password || undefined,
  };
}

// PROXY_LIST="host1:port:user:pass,host2:port:user:pass,..." seklinde virgulle ayrilmis havuzu okur.
function loadProxyPool() {
  if (!process.env.PROXY_LIST) return [];
  return process.env.PROXY_LIST
    .split(',')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .map(parseProxyEntry);
}

const PROXY_POOL = loadProxyPool();
let rotationIndex = 0;

// PROXY_LIST tanimliysa havuzdan sirayla (round-robin) bir sonraki proxy'yi doner - her cagri
// havuzu bir adim ilerletir, boylece art arda gelen cagirimlar farkli statik IP'lere duser.
// PROXY_LIST yoksa eski tekil PROXY_HOST/PORT/USERNAME/PASSWORD davranisina, o da yoksa
// proxysiz (dogrudan) baglantiya duser.
function getProxyConfig() {
  if (PROXY_POOL.length > 0) {
    const proxy = PROXY_POOL[rotationIndex % PROXY_POOL.length];
    rotationIndex += 1;
    return proxy;
  }

  if (!process.env.PROXY_HOST) {
    return null;
  }

  return {
    server: `${process.env.PROXY_HOST}:${process.env.PROXY_PORT}`,
    username: process.env.PROXY_USERNAME || undefined,
    password: process.env.PROXY_PASSWORD || undefined,
  };
}

module.exports = { getProxyConfig, loadProxyPool };
