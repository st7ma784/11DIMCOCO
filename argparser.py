from test_tube import HyperOptArgumentParser

class parser(HyperOptArgumentParser):
    def __init__(self,*args,strategy="grid_search"):

        super().__init__( *args,strategy=strategy, add_help=False) # or random search
        #more info at https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
        self.add_argument("--dir",default="/nobackup/projects/bdlan05/$USER/data",type=str)
        self.add_argument("--log_path",default="/nobackup/projects/bdlan05/$USER/logs/",type=str)
        self.opt_list("--learning_rate", default=0.001, type=float, options=[2e-4,1e-4,5e-5,1e-5], tunable=True)
        self.opt_list("--batch_size", default=16, type=int, options=[16, 20,], tunable=True)
        self.opt_list("--precision", default=16, options=[16,32], tunable=False)
        self.opt_list("--CaptioningPolicy", default="Individual", options=["Individual","RandomIndividual","All"], tunable=False)
        self.opt_list("--UsePreTrainedIm", default=False, type=bool, options=[True,False], tunable=False)
        self.opt_list("--UsePreTrainedEn", default=False, type=bool, options=[True,False], tunable=False)
        self.opt_list("--UseSeparateEncoder", default=True, type=bool, options=[True,False], tunable=False)
      
if __name__== "__main__":
    myparser=parser()
    hyperparams = myparser.parse_args()
    print(hyperparams.__dict__)
    for trial in hyperparams.trials(num=10):
        print(trial)
        
