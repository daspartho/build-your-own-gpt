export interface AIBlock {
  id: number;
  name: string;
  value: number | string;
  status: Status;
  color: string;
  hasInput: boolean;
  label?: string;
  isDone: boolean;
}

export enum Status {
  Panel,
  Model,
  Done,
}

export enum ComponentStatus {
  Panel = "Panel",
  Model = "Model",
}
