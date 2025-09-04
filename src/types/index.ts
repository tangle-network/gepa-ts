export interface DataInst {
  [key: string]: any;
}

export interface Trajectory {
  [key: string]: any;
}

export interface RolloutOutput {
  [key: string]: any;
}

export type ComponentMap = Record<string, string>;

export interface ReflectiveDataset {
  [componentName: string]: Array<Record<string, any>>;
}

export interface ProposalResult {
  candidate: ComponentMap;
  tag: 'reflective' | 'merge';
  parentProgramIds: number[];
  subsampleScoresBefore?: number[];
  subsampleScoresAfter?: number[];
}

export interface LoggerProtocol {
  log(message: string): void;
}

export type LanguageModel = (prompt: string) => Promise<string> | string;

export interface WandBConfig {
  apiKey?: string;
  initKwargs?: Record<string, any>;
}