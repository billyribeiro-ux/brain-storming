// The cockpit is an app, not a document: client-rendered, never prerendered.
// All data flows through TanStack Query + the WebSocket feed at runtime.
export const ssr = false;
export const prerender = false;
