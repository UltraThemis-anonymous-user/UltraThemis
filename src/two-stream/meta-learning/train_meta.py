import  torch, os
import  numpy as np
from    GetMetaDataSet import MetaDataSet
from    torch.utils.data import DataLoader
import  argparse
from meta import Meta
import config as cfg
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def main():

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    print(args)


    device = torch.device('cuda')
    maml = Meta(args).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    # enroll_name_list should be provided, which will be used as new user(only used in test)
    metadata = MetaDataSet([cfg.normal_data_train, cfg.nodfs_data_train], mode="train", batch_num=10000, person_num=args.n_way, k_support=args.k_spt, k_query=args.k_qry, enroll_name_list=[])
    metadata_test = MetaDataSet([cfg.normal_data_test, cfg.nodfs_data_test], mode="test", batch_num=100, person_num=args.n_way, k_support=args.k_spt, k_query=args.k_qry, enroll_name_list=["newuser1", "newuser2"])
    for epoch in range(args.epoch//10000):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(metadata, args.task_num, shuffle=True)
        loss_total = 0
        accs_total = 0
        for step, (support_ultra, support_voice, support_y, query_ultra, query_voice, query_y) in enumerate(db):

            support_ultra, support_voice, support_y, query_ultra, query_voice, query_y = \
                    support_ultra.to(device), support_voice.to(device), support_y.to(device), query_ultra.to(device), query_voice.to(device), query_y.to(device)

            accs, loss = maml(support_ultra, support_voice, support_y, query_ultra, query_voice, query_y)

            loss_total+=loss
            accs_total+=accs[-1]
            print("Step: {}/{} | loss: {:.4f}, Acc:{:.4f}".format(step+1, len(db), loss, accs[-1]), end='\r')
            if (step+1) % 100 == 0: # evaluation   
                print("Epoch: {}/{} | Step: {}/{} | train loss: {:.4f} | acc: {:.4f}".format(epoch+1, args.epoch//10000, step+1, len(db), loss_total/100, accs_total/100))
                loss_total = 0
                accs_total = 0
                db_test = DataLoader(metadata_test, 1, shuffle=False) 

                for step2, (support_ultra, support_voice, support_y, query_ultra, query_voice, query_y) in enumerate(db_test):
                    support_ultra, support_voice, support_y, query_ultra, query_voice, query_y = \
                    support_ultra.to(device), support_voice.to(device), support_y.to(device), query_ultra.to(device), query_voice.to(device), query_y.to(device)

                    accs, loss = maml.finetunning(support_ultra[0], support_voice[0], support_y[0], query_ultra[0], query_voice[0], query_y[0])
                    loss_total+=loss
                    accs_total+=accs[-1]
                    print("Step: {}/{} | loss: {:.4f}, Acc:{:.4f}".format(step2+1, len(db_test), loss, accs[-1]), end='\r')
                # [b, update_step+1]
                print("Epoch: {}/{} | Step: {}/{} | test loss: {:.4f} | acc: {:.4f}".format(epoch+1, args.epoch//10000, step+1, len(db), loss_total/len(db_test), accs_total/len(db_test)))
                loss_total = 0
                accs_total = 0
                maml.save_model(args.model_save_path)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_save_path', type=str, help='the path where the model is saved')
    argparser.add_argument('--epoch', type=int, help='epoch number', default=60000)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=4)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--batch_size', type=int, default=8)
    argparser.add_argument('--threshold', type=float, default=0.5)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()
    main()
