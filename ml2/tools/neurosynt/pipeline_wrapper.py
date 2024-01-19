from datetime import timedelta
from functools import cmp_to_key
from typing import Generator, List, Optional, Tuple

from ml2.datasets import GeneratorDataset
from ml2.ltl.ltl_mc import LTLMCSolution
from ml2.ltl.ltl_spec.decomp_ltl_spec import DecompLTLSpec
from ml2.ltl.ltl_syn import TFSynHierTransformerPipeline
from ml2.ltl.ltl_syn.ltl_syn_problem import LTLSynSolution
from ml2.ltl.ltl_syn.ltl_syn_status import LTLSynStatus
from ml2.pipelines import BeamSearchVerificationPipeline
from ml2.pipelines.samples import VerifiedBeamSearchSample
from ml2.tools.ltl_tool.tool_ltl_mc_problem import ToolLTLMCSolution
from ml2.tools.ltl_tool.tool_ltl_syn_problem import ToolLTLSynSolution


class PipelineWrapper:
    def __init__(
        self,
        beam_size: int,
        model: str,
        mc_port: int,
        verifier: str,
        batch_size: int,
        alpha: float,
        num_properties: int,
        length_properties: int,
    ) -> None:
        self.beam_size = beam_size
        self.model = model
        self.mc_port = mc_port
        self.batch_size = batch_size

        pipeline_config = {
            "type": "BeamSearchVerificationPipeline",
            "verifier": {
                "type": verifier,
                "start_containerized_service": False,
                "start_service": False,
                "port": mc_port,
            },
            "pipeline": {
                "base": "ltl-syn/" + model + "/train/pipe",
                "model_config": {"alpha": alpha, "beam_size": beam_size},
                "input_tokenizer": {
                    "base": "ltl-syn/" + model + "/train/pipe/input-tokenizer",
                    "num_props": num_properties,
                    "rename_aps_random": True,
                    "prop_tokenizer": {
                        "base": "ltl-syn/" + model + "/train/pipe/input-tokenizer/sub-tokenizer",
                        "pad": length_properties,
                    },
                },
                "max_local_num": num_properties,
                "max_local_length": length_properties,
            },
        }

        self.vpipeline: BeamSearchVerificationPipeline = BeamSearchVerificationPipeline.from_config(pipeline_config)  # type: ignore

        self.tpipeline: TFSynHierTransformerPipeline = self.vpipeline.pipeline  # type: ignore

    def eval_sample(
        self, sample: DecompLTLSpec, allow_unsound: bool = False, smallest_result: bool = True
    ) -> Tuple[ToolLTLSynSolution, Optional[ToolLTLMCSolution]]:
        result: VerifiedBeamSearchSample = self.vpipeline.eval_sample(sample)
        return self.filter_nonsuccess(
            sample=result, allow_unsound=allow_unsound, smallest_result=smallest_result
        )

    def filter_nonsuccess(
        self, sample: VerifiedBeamSearchSample, allow_unsound: bool = False, smallest_result=True
    ) -> Tuple[ToolLTLSynSolution, Optional[ToolLTLMCSolution]]:
        verified_results: List[
            Tuple[
                Optional[LTLSynSolution], Optional[float], Optional[LTLMCSolution], Optional[float]
            ]
        ] = [
            (beam.pred, beam.time, beam.verification, beam.verification_time)
            for beam in filter(
                lambda beam: beam.verification is not None
                and (beam.verification.validation_success or allow_unsound),
                sample.beams,
            )
        ]
        predictions = []
        for pred, syn_time, ver, ver_time in verified_results:
            assert (
                pred is not None
                and ver is not None
                and syn_time is not None
                and ver_time is not None
                and ver.tool is not None
            )
            predictions.append(
                (
                    ToolLTLSynSolution(
                        status=pred.status,
                        detailed_status="",
                        tool="NeuroSynt",
                        time=timedelta(seconds=syn_time),
                        circuit=pred.circuit,
                        realizable=pred.status.realizable,
                    ),
                    ToolLTLMCSolution(
                        status=ver.status,
                        detailed_status="",
                        tool=ver.tool,
                        time=timedelta(seconds=ver_time),
                        counterexample=ver.trace,
                    ),
                )
            )
        smallest: Optional[Tuple[ToolLTLSynSolution, ToolLTLMCSolution]] = self.find_smallest(
            predictions, smallest=smallest_result
        )
        if smallest is None:
            result: Tuple[ToolLTLSynSolution, Optional[ToolLTLMCSolution]] = (
                ToolLTLSynSolution(
                    status=LTLSynStatus("nonsuccess"),
                    detailed_status="No (correct) circuit was found.",
                    tool="NeuroSynt",
                    time=None,
                ),
                None,
            )
            return result
        else:
            return smallest

    @staticmethod
    def find_smallest(
        predictions: List[Tuple[ToolLTLSynSolution, ToolLTLMCSolution]], smallest: bool = True
    ) -> Optional[Tuple[ToolLTLSynSolution, ToolLTLMCSolution]]:
        def compare(item1, item2):
            dif = item1[0].circuit.num_latches - item2[0].circuit.num_latches
            if dif != 0:
                return dif
            else:
                return item1[0].circuit.num_ands - item2[0].circuit.num_ands

        if len(predictions) == 0:
            return None
        else:
            if smallest:
                predictions = sorted(predictions, key=cmp_to_key(compare))
            return predictions[0]

    def eval_batch(
        self, sample_generator: Generator[DecompLTLSpec, None, None]
    ) -> Generator[
        Tuple[DecompLTLSpec, ToolLTLSynSolution, Optional[ToolLTLMCSolution]], None, None
    ]:
        dataset = GeneratorDataset(
            name="gen_data", dtype=DecompLTLSpec, generator=sample_generator
        )
        samples, errs = self.tpipeline.eval(dataset=dataset, batch_size=self.batch_size)

        for sample in errs + samples:
            verified_sample = self.vpipeline.verify_sample(sample)
            result = self.filter_nonsuccess(verified_sample)
            yield (sample.inp, result[0], result[1])
