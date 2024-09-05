
from train import train_RGCN
from test import (test_RGCN,test_RGCN_hit_ByLength,test_RGCN_map,test_RGCN_map_ByLength,test_RGCN_Recall_ByLength_in2,
                  test_RGCN_map_ByLength_in2,test_RGCN_Recall_in2)


pattern_path = 'pattern_transition_matrix.xlsx'
cve_path = 'cve_transition_matrix.xlsx'
casual_path = 'causal_transition_matrix.xlsx'
model_name = "RGCN_Decoder_causal_29w.pth"
data_path = "out_2_train.csv"
test_path = "out_2_test.csv"
model, graph = train_RGCN(num_nodes=123, h_dim=64, out_dim=16, num_rels=3, num_epochs=10, batch=1000
                          , pattern_path=pattern_path, cve_path=cve_path, casual_path=casual_path,
                          data_path=data_path, model_name=model_name)

test_RGCN(model=model, graph=graph, test_path=test_path)
