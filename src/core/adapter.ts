import { ComponentMap, DataInst, ReflectiveDataset, RolloutOutput, Trajectory } from '../types/index.js';

export interface EvaluationBatch<T = Trajectory, O = RolloutOutput> {
  outputs: O[];
  scores: number[];
  trajectories?: T[] | null;
}

export type ProposalFn = (
  candidate: ComponentMap,
  reflectiveDataset: ReflectiveDataset,
  componentsToUpdate: string[]
) => ComponentMap | Promise<ComponentMap>;

export interface GEPAAdapter<D = DataInst, T = Trajectory, O = RolloutOutput> {
  evaluate(
    batch: D[],
    candidate: ComponentMap,
    captureTraces?: boolean
  ): EvaluationBatch<T, O> | Promise<EvaluationBatch<T, O>>;

  makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch<T, O>,
    componentsToUpdate: string[]
  ): ReflectiveDataset | Promise<ReflectiveDataset>;

  proposeNewTexts?: ProposalFn;
}

export abstract class BaseAdapter<D = DataInst, T = Trajectory, O = RolloutOutput>
  implements GEPAAdapter<D, T, O>
{
  abstract evaluate(
    batch: D[],
    candidate: ComponentMap,
    captureTraces?: boolean
  ): EvaluationBatch<T, O> | Promise<EvaluationBatch<T, O>>;

  abstract makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch<T, O>,
    componentsToUpdate: string[]
  ): ReflectiveDataset | Promise<ReflectiveDataset>;

  proposeNewTexts?: ProposalFn;
}