/// <reference types="vite/client" />

declare module 'virtual:flexweb-cars' {
  export interface CarModelEntry {
    id: string;
    name: string;
    url: string;
    bytes: number;
    mtimeMs?: number;
  }
  const cars: CarModelEntry[];
  export default cars;
}
