export const config = {
  api: {
    bodyParser: false
  }
};

const bodyMethods = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

async function readBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(Buffer.from(chunk));
  }
  return Buffer.concat(chunks);
}

export default async function handler(request, response) {
  const backendUrl = process.env.BACKEND_URL;
  const path = Array.isArray(request.query.path)
    ? request.query.path[0]
    : request.query.path || '/';

  if (!backendUrl) {
    response.status(500).json({ error: 'BACKEND_URL is not configured' });
    return;
  }

  const target = new URL(path, `${backendUrl.replace(/\/$/, '')}/`);
  const headers = new Headers();
  for (const [name, value] of Object.entries(request.headers)) {
    if (name.toLowerCase() !== 'host' && name.toLowerCase() !== 'content-length') {
      headers.set(name, Array.isArray(value) ? value.join(',') : value);
    }
  }

  const body = bodyMethods.has(request.method) ? await readBody(request) : undefined;
  const upstream = await fetch(target, {
    method: request.method,
    headers,
    body
  });

  response.status(upstream.status);
  upstream.headers.forEach((value, name) => {
    response.setHeader(name, value);
  });
  response.send(Buffer.from(await upstream.arrayBuffer()));
}
