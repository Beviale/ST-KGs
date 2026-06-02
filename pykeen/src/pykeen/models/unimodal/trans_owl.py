"""TransOWL: TransE with OWL/RDFS axiom-based regularization on relations."""

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

from class_resolver import Hint, HintOrType, OptionalKwargs
from torch.nn import functional

from ..nbase import ERModel
from ...constants import DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE
from ...nn import TransEInteraction
from ...nn.init import xavier_uniform_, xavier_uniform_norm_
from ...regularizers import AxiomRegularizer, Regularizer
from ...typing import Constrainer, FloatTensor, Initializer

__all__ = [
    "TransOWL",
]


class TransOWL(ERModel[FloatTensor, FloatTensor, FloatTensor]):
    r"""TransOWL [damato2021]_: TransE scoring with axiom-based regularization.

    The scoring function is identical to :class:`~pykeen.models.TransE`, i.e.
    $f(h, r, t) = -\lVert h + r - t\rVert_p$. Background knowledge (BK) is injected by
    regularizing the relation embeddings *directly* via :class:`~pykeen.regularizers.AxiomRegularizer`:

    - ``owl:inverseOf`` pairs ($r \equiv q^-$) are pushed towards opposite vectors ($\lVert r + q\rVert$);
    - ``owl:equivalentProperty`` pairs ($r \equiv p$) are pushed towards equal vectors ($\lVert r - p\rVert$).

    The axiom pairs are provided as relation indices, matching the
    :class:`~pykeen.triples.TriplesFactory` ``relation_to_id`` mapping. If no pairs are
    provided, the model is equivalent to :class:`~pykeen.models.TransE`.
    ---
    citation:
        author: d'Amato
        year: 2021
        link: https://doi.org/10.1007/978-3-030-91305-2_1
    """

    #: The default strategy for optimizing the model's hyper-parameters
    hpo_default: ClassVar[Mapping[str, Any]] = {
        "embedding_dim": DEFAULT_EMBEDDING_HPO_EMBEDDING_DIM_RANGE,
        "scoring_fct_norm": {"type": int, "low": 1, "high": 2},
    }

    def __init__(
        self,
        *,
        embedding_dim: int = 50,
        scoring_fct_norm: int = 1,
        power_norm: bool = False,
        inverse_relations: Iterable[tuple[int, int]] | None = None,
        equivalent_relations: Iterable[tuple[int, int]] | None = None,
        subproperty_relations: Iterable[tuple[int, int]] | None = None,
        inverse_weight: float = 1.0,
        equivalence_weight: float = 1.0,
        subproperty_weight: float = 0.01,
        beta: float = 0.9,
        regularizer_weight: float = 1.0,
        entity_initializer: Hint[Initializer] = xavier_uniform_,
        entity_constrainer: Hint[Constrainer] = functional.normalize,
        relation_initializer: Hint[Initializer] = xavier_uniform_norm_,
        relation_constrainer: Hint[Constrainer] = None,
        relation_regularizer: HintOrType[Regularizer] = None,
        relation_regularizer_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        r"""Initialize TransOWL.

        :param embedding_dim: The entity/relation embedding dimension $d$.
        :param scoring_fct_norm:
            The norm used with :func:`torch.linalg.vector_norm`. Typically 1 or 2.
        :param power_norm:
            Whether to use the p-th power of the $L_p$ norm.

        :param inverse_relations:
            relation index pairs $(r, q)$ such that $r \equiv q^-$ (``owl:inverseOf``).
        :param equivalent_relations:
            relation index pairs $(r, p)$ such that $r \equiv p$ (``owl:equivalentProperty``).
        :param subproperty_relations:
            relation index pairs $(r, p)$ such that $r \sqsubseteq p$ (``rdfs:subPropertyOf``);
            order matters ($r$ sub-property, $p$ super-property).
        :param inverse_weight:
            weight $\lambda_1$ for the inverseOf regularization term.
        :param equivalence_weight:
            weight $\lambda_2$ for the equivalentProperty regularization term.
        :param subproperty_weight:
            weight for the subPropertyOf regularization term.
        :param beta:
            directional offset of the subPropertyOf term; $\beta = 1$ reduces it to plain
            equality $\lVert r - p\rVert$.
        :param regularizer_weight:
            the overall axiom regularization weight $\lambda$.

        :param entity_initializer: Entity initializer. Defaults to :func:`pykeen.nn.init.xavier_uniform_`.
        :param entity_constrainer: Entity constrainer. Defaults to :func:`torch.nn.functional.normalize`.
        :param relation_initializer:
            Relation initializer. Defaults to :func:`pykeen.nn.init.xavier_uniform_norm_`.
        :param relation_constrainer: Relation constrainer. Defaults to none.

        :param relation_regularizer:
            Override the axiom regularizer with a custom one. By default,
            :class:`~pykeen.regularizers.AxiomRegularizer` is used with the axiom pairs above.
        :param relation_regularizer_kwargs:
            keyword-based parameters for a custom ``relation_regularizer``.

        :param kwargs:
            Remaining keyword arguments forwarded to :class:`~pykeen.models.ERModel`.
        """
        super().__init__(
            interaction=TransEInteraction,
            interaction_kwargs={"p": scoring_fct_norm, "power_norm": power_norm},
            entity_representations_kwargs={
                "shape": embedding_dim,
                "initializer": entity_initializer,
                "constrainer": entity_constrainer,
            },
            relation_representations_kwargs={
                "shape": embedding_dim,
                "initializer": relation_initializer,
                "constrainer": relation_constrainer,
            },
            **kwargs,
        )

        # The axiom-based regularization operates on the *whole* relation embedding matrix,
        # so we register a weight regularizer rather than a per-batch Embedding regularizer
        # (mirroring the approach used by TransH).
        if relation_regularizer is None and relation_regularizer_kwargs is None:
            relation_regularizer = AxiomRegularizer
            relation_regularizer_kwargs = {
                "inverse_pairs": inverse_relations,
                "equivalence_pairs": equivalent_relations,
                "subproperty_pairs": subproperty_relations,
                "inverse_weight": inverse_weight,
                "equivalence_weight": equivalence_weight,
                "subproperty_weight": subproperty_weight,
                "beta": beta,
                "p": scoring_fct_norm,
                "weight": regularizer_weight,
            }
        self.append_weight_regularizer(
            parameter=self.relation_representations[0].parameters(),
            regularizer=relation_regularizer,
            regularizer_kwargs=relation_regularizer_kwargs,
        )
