# MODEL OPTS
def model_args(parser):
    group = parser.add_argument_group('Model', 'Arguments control Model')
    group.add_argument('--arch', default='ResNet', type=str, choices=['ResNet'], 
                       help='model architecture')
    group.add_argument('--depth', default=20, type=int, 
                       help='depth of the model')
    group.add_argument('--model-num', default=3, type=int, 
                       help='number of submodels within the ensemble')
    group.add_argument('--model-file', default=None, type=str,
                       help='Path to the file that contains model checkpoints')
    group.add_argument('--gpu', default='0', type=str, 
                       help='gpu id')
    group.add_argument('--seed', default=0, type=int,
                       help='random seed for torch')


# DATALOADING OPTS
def data_args(parser):
    group = parser.add_argument_group('Data', 'Arguments control Data and loading for training')
    group.add_argument('--data-dir', type=str, default='./data',
                       help='Dataset directory')
    group.add_argument('--batch-size', type=int, default=128,
                       help='batch size of the train loader')


# BASE TRAINING ARGS
def base_train_args(parser):
    group = parser.add_argument_group('Base Training', 'Base arguments to configure training')
    group.add_argument('--epochs', default=200, type=int, 
                       help='number of training epochs')
    group.add_argument('--lr', default=0.1, type=float, 
                       help='learning rate')
    group.add_argument('--sch-intervals', nargs='*', default=[100,150], type=int,
                       help='learning scheduler milestones')
    group.add_argument('--lr-gamma', default=0.1, type=float, 
                       help='learning rate decay ratio')


# DVERGE TRAINING ARGS
def dverge_train_args(parser):
    group = parser.add_argument_group('DVERGE Training', 'Arguments to configure DVERGE training')
    group.add_argument('--distill-eps', default=0.07, type=float, 
                       help='perturbation budget for distillation')
    group.add_argument('--distill-alpha', default=0.007, type=float, 
                       help='step size for distillation')
    group.add_argument('--distill-steps', default=10, type=int, 
                       help='number of steps for distillation')
    group.add_argument('--distill-fixed-layer', default=False, action="store_true",
                       help='whether fixing the layer for distillation')
    group.add_argument('--distill-layer', default=20, type=int, 
                       help='which layer is used for distillation, only useful when distill-fixed-layer is True')
    group.add_argument('--distill-rand-start', default=False, action="store_true",
                       help='whether use random start for distillation')
    group.add_argument('--distill-no-momentum', action="store_false", dest='distill_momentum',
                       help='whether use momentum for distillation')
    group.add_argument('--plus-adv', default=False, action="store_true",
                       help='whether perform adversarial training in the mean time with diversity training')
    group.add_argument('--dverge-coeff', default=1., type=float,
                       help='the coefficient to balance diversity training and adversarial training')
    group.add_argument('--start-from', default='baseline', type=str, choices=['baseline', 'scratch'],
                       help='starting point of the training')
    group.add_argument('--eps', default=8./255., type=float, 
                       help='perturbation budget for adversarial training')
    group.add_argument('--alpha', default=2./255., type=float, 
                       help='step size for adversarial training')
    group.add_argument('--steps', default=10, type=int, 
                       help='number of steps for adversarial training')
    

# ADVERSARIAL TRAINING ARGS
def adv_train_args(parser):
    group = parser.add_argument_group('Adversarial Training', 'Arguments to configure adversarial training')
    group.add_argument('--eps', default=8./255., type=float, 
                       help='perturbation budget for adversarial training')
    group.add_argument('--alpha', default=2./255., type=float, 
                       help='step size for adversarial training')
    group.add_argument('--steps', default=10, type=int, 
                       help='number of steps for adversarial training')


# ADP TRAINING ARGS
# https://arxiv.org/abs/1901.08846
def adp_train_args(parser):
    group = parser.add_argument_group('ADP Training', 'Arguments to configure ADP training')
    group.add_argument('--alpha', default=2.0, type=float, 
                       help='coefficient for ensemble entropy')
    group.add_argument('--beta', default=0.5, type=float, 
                       help='coefficient for log determinant')
    group.add_argument('--plus-adv', default=False, action="store_true",
                       help='whether perform adversarial training in the mean time with diversity training')
    group.add_argument('--adv-eps', default=8./255., type=float, 
                       help='perturbation budget for adversarial training')
    group.add_argument('--adv-alpha', default=2./255., type=float, 
                       help='step size for adversarial training')
    group.add_argument('--adv-steps', default=10, type=int, 
                       help='number of steps for adversarial training')


# GAL TRAINING ARGS
# https://arxiv.org/pdf/1901.09981.pdf
def gal_train_args(parser):
    group = parser.add_argument_group('GAL Training', 'Arguments to configure GAL training')
    group.add_argument('--lambda', default=.5, type=float, 
                       help='coefficient for coherence')
    group.add_argument('--plus-adv', default=False, action="store_true",
                       help='whether perform adversarial training in the mean time with diversity training')
    group.add_argument('--adv-eps', default=8./255., type=float, 
                       help='perturbation budget for adversarial training')
    group.add_argument('--adv-alpha', default=2./255., type=float, 
                       help='step size for adversarial training')
    group.add_argument('--adv-steps', default=10, type=int, 
                       help='number of steps for adversarial training')


# WBOX EVALUATION ARGS
def wbox_eval_args(parser):
    group = parser.add_argument_group('White-box Evaluation', 'Arguments to configure evaluation of white-box robustness')
    group.add_argument('--subset-num', default=1000, type=int, 
                       help='number of samples of the subset, will use the full test set if none')
    group.add_argument('--random-start', default=5, type=int, 
                       help='number of random starts for PGD')
    group.add_argument('--steps', default=50, type=int, 
                       help='number of steps for PGD')
    group.add_argument('--loss-fn', default='xent', type=str, choices=['xent', 'cw'],
                       help='which loss function to use')
    group.add_argument('--cw-conf', default=.1, type=float,
                       help='confidence for cw loss function')
    group.add_argument('--save-to-csv', action="store_true",
                       help='whether save the results to a csv file')
    group.add_argument('--overwrite', action="store_false", dest="append_out",
                       help='when saving results, whether use append mode')
    group.add_argument('--convergence-check', action="store_true", 
                       help='whether perform sanity check to make sure the attack converges')


# BBOX TRANSFER EVALUATION ARGS
def bbox_eval_args(parser):
    group = parser.add_argument_group('Black-box Evaluation', 'Arguments to configure evaluation of black-box robustness')
    group.add_argument('--folder', default='transfer_adv_examples', type=str, 
                       help='name of the folder that contains transfer adversarial examples')
    group.add_argument('--steps', default=100, type=int,
                       help='number of PGD steps for convergence check')
    group.add_argument('--which-ensemble', default='baseline', choices=['baseline', 'dverge', 'adp', 'gal'],
                       help='transfer from which ensemble')
    group.add_argument('--save-to-csv', action="store_true",
                       help='whether save the results to a csv file')
    group.add_argument('--overwrite', action="store_false", dest="append_out",
                       help='when saving results, whether use append mode')
                       

# TRANSFERABILITY EVALUATION ARGS
def transf_eval_args(parser):
    group = parser.add_argument_group('Transferability Evaluation', 'Arguments to configure evaluation of transferablity among submodels')
    group.add_argument('--subset-num', default=1000, type=int, 
                       help='number of samples of the subset')
    group.add_argument('--random-start', default=5, type=int, 
                       help='number of random starts for PGD')
    group.add_argument('--steps', default=50, type=int, 
                       help='number of steps for PGD')
    group.add_argument('--save-to-file', action="store_true",
                       help='whether save the results to a file')


# DIVERSITY EVALUATION ARGS
def diversity_eval_args(parser):
    group = parser.add_argument_group('Diversity Evaluation', 'Arguments to configure evaluation of diversity of the ensemble')
    group.add_argument('--subset-num', default=1000, type=int, 
                       help='number of samples of the subset')
    group.add_argument('--save-to-file', action="store_true",
                       help='whether save the results to a file')
    
