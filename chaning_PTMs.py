from CleverCopyPredictorBeforeFinetuning import CleverCopyPredictorBeforeFineTuning
from CleverCopyPredictorFinetuning import CleverCopyPredictorFineTuning

class Chaining_PTMs():

    def __init__(self,chain,path_data,save_path,base_prosit_weights,fragmentation = "HCD", trainsize="max",init_model_data_path=None):
        self.chain = list(chain)
        self.path_data = str(path_data)
        self.save_path = str(save_path)
        self.trainsize = str(trainsize)
        self.trainsize = trainsize
        self.fragmentation = fragmentation
        self.init_model_data_path = str(init_model_data_path)
        self.base_prosit_weights = str(base_prosit_weights)

        self.start_mod = chain[0]
        self.final_mod = chain[-1]

        self.chain_name = "-".join(self.chain)

        self.result_dict = {}

    def chaining(self):

        for mod in self.chain:
            if mod == self.start_mod:

                CC_predictor = CleverCopyPredictorBeforeFineTuning(
                    modification=mod,
                    test_path=f'{self.path_data}/test_{self.fragmentation}_{mod}_max.parquet',
                    val_path=f'{self.path_data}/val_{self.fragmentation}_{mod}_max.parquet',
                    case='chaining_start',
                    path_weights=self.base_prosit_weights,
                    save_path=self.save_path,
                )

                CC_predictor.prediction()
                amino_acid = CC_predictor.get_best_AA()
                SA_before_ft = CC_predictor.get_SA_before_ft()
                print(f"best AA before fine-tuning\t{amino_acid}: {SA_before_ft}")
                path_weights = CC_predictor.get_weights_path_of_best_AA()

                CC_predictor_finetune = CleverCopyPredictorFineTuning(
                    modification=mod,
                    train_path=f'{self.path_data}/train_{self.fragmentation}_{mod}_{self.trainsize}.parquet',
                    test_path=f'{self.path_data}/test_{self.fragmentation}_{mod}_max.parquet',
                    val_path=f'{self.path_data}/val_{self.fragmentation}_{mod}_max.parquet',
                    trainsize=self.trainsize,
                    case="chaining_start",
                    path_weights=path_weights,
                    save_path=self.save_path,
                    amino_acid=amino_acid
                )

                CC_predictor_finetune.train_model()
                CC_predictor_finetune.predict_and_save_results()
                path_weights_after_ft = CC_predictor_finetune.get_weights_path_after_ft()
                SA_after_ft = CC_predictor_finetune.get_SA_after_ft()
                print(f"SA after fine-tuning\t{amino_acid}: {SA_after_ft}")

                self.result_dict[mod] = {'before':{'AA':amino_acid, 'SA':SA_before_ft, 'path_weights':path_weights},
                                         'after':{'AA':amino_acid, 'SA': SA_after_ft, 'path_weights':path_weights_after_ft}}
            
            else:
                mods_to_add_to_alphabet = self.chain[0:self.chain.index(mod)]
                combined_train_set = self.chain[0:(self.chain.index(mod))+1]
                print(combined_train_set)
                combined_train_set = "-".join(combined_train_set)

                CC_predictor = CleverCopyPredictorBeforeFineTuning(
                    modification=mod,
                    test_path=f'{self.path_data}/test_{self.fragmentation}_{mod}_max.parquet',
                    val_path=f'{self.path_data}/val_{self.fragmentation}_{mod}_max.parquet',
                    case='chaining',
                    path_weights=path_weights_after_ft,
                    mods_to_add_to_alphabet = mods_to_add_to_alphabet,
                    save_path=self.save_path,
                )

                CC_predictor.prediction()
                amino_acid = CC_predictor.get_best_AA()
                SA_before_ft = CC_predictor.get_SA_before_ft()
                print(f"best AA before fine-tuning\t{amino_acid}: {SA_before_ft}")
                path_weights_before_ft = CC_predictor.get_weights_path_of_best_AA()

                CC_predictor_finetune = CleverCopyPredictorFineTuning(
                    modification=mod,
                    train_path=f'{self.path_data}/chaining/{self.chain_name}/train_{self.fragmentation}_{mod}_{combined_train_set}_{self.trainsize}.parquet',
                    test_path=f'{self.path_data}/test_{self.fragmentation}_{mod}_max.parquet',
                    val_path=f'{self.path_data}/val_{self.fragmentation}_{mod}_max.parquet',
                    trainsize=self.trainsize,
                    case="chaining",
                    path_weights=path_weights_before_ft,
                    save_path=self.save_path,
                    mods_to_add_to_alphabet = mods_to_add_to_alphabet,
                    amino_acid=amino_acid
                )
                
                CC_predictor_finetune.train_model()
                CC_predictor_finetune.predict_and_save_results()
                path_weights_after_ft = CC_predictor_finetune.get_weights_path_after_ft()
                SA_after_ft = CC_predictor_finetune.get_SA_after_ft()
                print(f"SA after fine-tuning\t{amino_acid}: {SA_after_ft}")

                self.result_dict[mod] = {'before':{'AA':amino_acid, 'SA':SA_before_ft, 'path_weights':path_weights},
                                    'after':{'AA':amino_acid, 'SA': SA_after_ft, 'path_weights':path_weights_after_ft}}

                if mod == self.final_mod:
                    self.summary()
                    print(f"\nWeights of final model of chaining approach after fine-tuning:\t{path_weights_after_ft}")
                
                
    def get_results(self):
        return self.result_dict
    
    def summary(self):
        print(f"Results of chaining: {', '.join(str(p) for p in self.chain) }")
        for i,mod in enumerate(self.chain,start=1):
            print(f"{i}. {mod}\tbefore ft:\t{self.result_dict[mod]['before']['AA']},\t{self.result_dict[mod]['before']['SA']}")
            print(f"   {mod}\tafter ft:\t{self.result_dict[mod]['after']['AA']},\t{self.result_dict[mod]['after']['SA']}")