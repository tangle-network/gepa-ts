/**
 * Core types for GEPA optimization system
 */

export interface GEPACandidate {
  prompt?: string;
  [key: string]: any;
}

export interface GEPAResult {
  accuracy: number;
  scores: number[];
  predictions?: any[];
  metadata?: {
    duration?: number;
    totalExamples?: number;
    successfulPredictions?: number;
    [key: string]: any;
  };
}

export interface ComponentMap {
  [componentName: string]: string;
}

export interface EvaluationBatch<TTrajectory = any, TOutput = any> {
  outputs: TOutput[];
  scores: number[];
  trajectories?: TTrajectory[] | null;
}

export interface DataInst {
  [key: string]: any;
}

export interface Trajectory {
  [key: string]: any;
}

export interface RolloutOutput {
  [key: string]: any;
}

export interface ReflectiveDataset {
  [componentName: string]: Array<Record<string, any>>;
}

// Export commonly used types
export type { GEPACandidate as Candidate };
export type { GEPAResult as Result };
export type { ComponentMap as Components };