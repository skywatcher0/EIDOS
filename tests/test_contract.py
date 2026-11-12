import unittest
import numpy as np
from eidos.protocol.strict import five_role_split_by_env
from eidos.mdl.data_code import UniformQuantizer, bin_prob_mass
from eidos.dsl.grammar import Const, Var
from eidos.search.models import CandidateSpec, LinearBasisModel
from eidos.serialization import pack, unpack
from eidos._io import canonical_bytes

class ContractTests(unittest.TestCase):
    def test_roles_cover_once_and_are_deterministic(self):
        env=np.repeat(np.arange(5),60)
        first=five_role_split_by_env(env,seed=42)
        self.assertTrue(np.array_equal(first,five_role_split_by_env(env,seed=42)))
        self.assertEqual(set(first),{'disc','diag','audit','cal','test'})
        self.assertEqual(sum(np.count_nonzero(first==r) for r in set(first)),300)

    def test_half_open_quantizer_boundaries(self):
        q=UniformQuantizer(delta=2,origin=1)
        values=np.array([-2.,0.,1.,2.,4.])
        index=q.index(values)
        self.assertEqual(index.tolist(),[-1,0,0,1,2])
        lo,hi=q.bounds(index)
        self.assertTrue(np.all(values>=lo) and np.all(values<hi))

    def test_finite_bin_probability_normalization(self):
        y=np.arange(-4000,4001)*0.25
        probabilities=bin_prob_mass(y_tilde=y,mu=np.zeros_like(y),sigma=np.ones_like(y),delta_y=0.25,
                                   noise='student_t',df=3.0,robust_eps=0.02,escape_rho=0.98)
        self.assertAlmostEqual(float(probabilities.sum()),1.0,places=6)

    def test_decoder_has_all_prediction_state(self):
        x=np.linspace(-2,3,30).reshape(-1,1)
        spec=CandidateSpec('unit',[Const(1),Var('x0')],ridge=0.01)
        model=LinearBasisModel(spec,['x0'],ridge=0.01).fit(x,2*x[:,0]+0.3)
        model.set_param_vector(np.rint(model.param_vector()/0.001)*0.001)
        record=pack(model,0.001,{'selection_keys':['0x1.0p+0']})
        decoded=unpack(record)
        for before,after in zip(model.predict_mu_sigma(x),decoded.predict_mu_sigma(x)):
            self.assertTrue(np.array_equal(before,after))
        self.assertEqual(canonical_bytes(record),canonical_bytes(pack(decoded,0.001,{'selection_keys':['0x1.0p+0']})))

    def test_decoder_rejects_unknown_schema(self):
        with self.assertRaises(ValueError): unpack({'schema':'unknown'})

if __name__=='__main__': unittest.main()
