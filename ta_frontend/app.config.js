const { execSync } = require('node:child_process');
const os = require('node:os');
const appJson = require('./app.json');

const ANDROID_EMULATOR_HOST = '10.0.2.2';
const ANDROID_EMULATOR_API_BASE_URL = `http://${ANDROID_EMULATOR_HOST}:8000/api`;

function isPrivateLanIpv4(ip) {
  const match = ip.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (!match) {
    return false;
  }

  const octets = match.slice(1).map(Number);
  if (octets.some((value) => Number.isNaN(value) || value < 0 || value > 255)) {
    return false;
  }

  const [a, b] = octets;
  return a === 10 || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168);
}

function parseIpconfigForLanIp(output) {
  const blocks = output.split(/\r?\n\r?\n/);
  const disallowedAdapterName =
    /zerotier|vethernet|vmware|virtualbox|hyper-v|loopback|bluetooth|local area connection\*/i;

  let bestMatch;

  for (const block of blocks) {
    if (!/adapter/i.test(block) || /media disconnected/i.test(block)) {
      continue;
    }

    const headerLine = block.split(/\r?\n/)[0] ?? '';
    if (disallowedAdapterName.test(headerLine)) {
      continue;
    }

    const ipMatch = block.match(/IPv4 Address[^:]*:\s*(\d{1,3}(?:\.\d{1,3}){3})/i);
    const ip = ipMatch?.[1];

    if (!ip || !isPrivateLanIpv4(ip)) {
      continue;
    }

    const wifiPriority = /wi-?fi|wireless/i.test(headerLine) ? 2 : 1;
    if (!bestMatch || wifiPriority > bestMatch.priority) {
      bestMatch = { ip, priority: wifiPriority };
    }
  }

  return bestMatch?.ip;
}

function getLanIpFromIpconfig() {
  if (process.platform !== 'win32') {
    return undefined;
  }

  try {
    const output = execSync('ipconfig', { encoding: 'utf8' });
    return parseIpconfigForLanIp(output);
  } catch {
    return undefined;
  }
}

function getLanIpFromNetworkInterfaces() {
  const interfaces = os.networkInterfaces();
  const disallowedInterfaceName = /zerotier|vethernet|vmware|virtualbox|hyper-v|loopback|bluetooth/i;

  let ethernetIp;

  for (const [interfaceName, addresses] of Object.entries(interfaces)) {
    if (!addresses || disallowedInterfaceName.test(interfaceName)) {
      continue;
    }

    for (const addr of addresses) {
      if (addr.family !== 'IPv4' || addr.internal || !isPrivateLanIpv4(addr.address)) {
        continue;
      }

      if (/wi-?fi|wlan|wireless/i.test(interfaceName)) {
        return addr.address;
      }

      if (!ethernetIp) {
        ethernetIp = addr.address;
      }
    }
  }

  return ethernetIp;
}

const computedLanIp = getLanIpFromIpconfig() ?? getLanIpFromNetworkInterfaces();
const devTarget = process.env.EXPO_DEV_TARGET;
const explicitApiBaseUrl = process.env.EXPO_PUBLIC_API_BASE_URL ?? process.env.API_BASE_URL;
const useAndroidEmulatorTarget = devTarget === 'android-emulator';
const existingApiBaseUrl = appJson.expo?.extra?.apiBaseUrl;
const apiBaseUrl =
  explicitApiBaseUrl ??
  (useAndroidEmulatorTarget
    ? ANDROID_EMULATOR_API_BASE_URL
    : computedLanIp
      ? `http://${computedLanIp}:8000/api`
      : existingApiBaseUrl);
const backendHostIp = useAndroidEmulatorTarget ? ANDROID_EMULATOR_HOST : computedLanIp;

module.exports = {
  ...appJson,
  expo: {
    ...appJson.expo,
    plugins: [
      ...(appJson.expo?.plugins ?? []),
      '@react-native-community/datetimepicker',
    ],
    extra: {
      ...(appJson.expo?.extra ?? {}),
      apiBaseUrl,
      backendHostIp,
      devTarget,
    },
  },
};
