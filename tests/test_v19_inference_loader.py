import unittest
import torch
from chess_zero.model import AlphaZeroNet, load_inference_weights

class InferenceLoaderTests(unittest.TestCase):
    def test_unused_heads_do_not_change_inference(self):
        torch.set_num_threads(1)
        def net(): return AlphaZeroNet(blocks=1,channels=8,planes=30,se_ratio=4)
        source=net().eval()
        prefixes=("sm_","ss_","av_fc.")
        old={k:v for k,v in source.state_dict().items() if not k.startswith(prefixes)}
        loaded=load_inference_weights(net(),old).eval()
        x=torch.randn(2,30,8,8)
        with torch.no_grad():
            for a,b in zip(source.forward_inf(x),loaded.forward_inf(x)):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
        broken=dict(old);broken.pop("p_fc.weight")
        with self.assertRaises(RuntimeError): load_inference_weights(net(),broken)
        broken=dict(old);broken["v_fc2.weight"]=torch.zeros(1,1)
        with self.assertRaises(RuntimeError): load_inference_weights(net(),broken)

if __name__=="__main__": unittest.main()
