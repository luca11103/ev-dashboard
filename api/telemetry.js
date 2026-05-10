/* global process */

import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

let latestTelemetry = null;
let latestUpdatedAt = null;
const cacheFile = join(tmpdir(), 'ev-dashboard-latest-telemetry.json');

const json = (response, status, body) => {
  response.statusCode = status;
  response.setHeader('Content-Type', 'application/json');
  response.setHeader('Access-Control-Allow-Origin', '*');
  response.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  response.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-pico-key');
  response.end(JSON.stringify(body));
};

const readBody = (request) =>
  new Promise((resolve, reject) => {
    let body = '';
    request.on('data', (chunk) => {
      body += chunk;
      if (body.length > 1024 * 64) {
        request.destroy();
        reject(new Error('Payload too large'));
      }
    });
    request.on('end', () => resolve(body));
    request.on('error', reject);
  });

const loadLatestTelemetry = async () => {
  if (latestTelemetry) {
    return { data: latestTelemetry, updatedAt: latestUpdatedAt };
  }

  try {
    const cached = JSON.parse(await readFile(cacheFile, 'utf8'));
    latestTelemetry = cached.data || null;
    latestUpdatedAt = cached.updatedAt || null;
  } catch {
    latestTelemetry = null;
    latestUpdatedAt = null;
  }

  return { data: latestTelemetry, updatedAt: latestUpdatedAt };
};

const saveLatestTelemetry = async () => {
  await writeFile(
    cacheFile,
    JSON.stringify({ data: latestTelemetry, updatedAt: latestUpdatedAt }),
    'utf8',
  );
};

export default async function handler(request, response) {
  if (request.method === 'OPTIONS') {
    return json(response, 204, {});
  }

  if (request.method === 'GET') {
    const latest = await loadLatestTelemetry();
    return json(response, 200, {
      ok: true,
      updatedAt: latest.updatedAt,
      data: latest.data,
    });
  }

  if (request.method !== 'POST') {
    return json(response, 405, { ok: false, error: 'Method not allowed' });
  }

  const expectedKey = process.env.PICO_API_KEY;
  if (expectedKey && request.headers['x-pico-key'] !== expectedKey) {
    return json(response, 401, { ok: false, error: 'Invalid Pico API key' });
  }

  try {
    const rawBody = await readBody(request);
    const telemetry = JSON.parse(rawBody || '{}');

    latestUpdatedAt = new Date().toISOString();
    latestTelemetry = {
      ...telemetry,
      receivedAt: latestUpdatedAt,
    };
    await saveLatestTelemetry();

    return json(response, 200, {
      ok: true,
      updatedAt: latestUpdatedAt,
    });
  } catch (error) {
    return json(response, 400, {
      ok: false,
      error: error.message || 'Invalid telemetry payload',
    });
  }
}
