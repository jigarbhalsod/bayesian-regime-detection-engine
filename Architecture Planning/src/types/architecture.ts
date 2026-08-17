export type ArchitectureType =
  | "system"
  | "module"
  | "component"
  | "entity"
  | "procedure"
  | "method"
  | "workflow";

export type TriggerType =
  | "event-driven"
  | "time-driven"
  | "request-driven"
  | "batch"
  | "manual"
  | "milestone";

export type ArchitectureStatus = "MVP" | "Future" | "Both";

export interface ArchitectureItem {
  id: string;
  name: string;
  type: ArchitectureType;
  layer: string;
  description: string;

  inputs?: string[];
  outputs?: string[];

  modules?: string[];
  entities?: string[];
  procedures?: string[];
  methods?: string[];

  dependencies?: string[];

  technology?: string[];
  trigger?: TriggerType;
  frequency?: string;

  status?: ArchitectureStatus;

  security?: string[];
  failureHandling?: string[];

  children?: string[];
}

export interface ArchitectureEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
}

export interface ArchitectureLayer {
  id: string;
  name: string;
  description: string;
  color: string;
}

export interface ArchitectureData {
  items: ArchitectureItem[];
  edges: ArchitectureEdge[];
  layers: ArchitectureLayer[];
}