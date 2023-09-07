import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
import sys
from meta_model import ResBlock, ResNet
from copy import deepcopy
import learn2learn as L2L

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.person_num = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.batch_size = args.batch_size
        self.threshold = args.threshold
        self.net = ResNet(ResBlock)
        
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)




    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter

    def compute_loss(self, model, ultra_list, voice_list, y_list, vars=None):
        """
        :param x_spt:   [supportsz, c_, f, t]
        :param y_spt:   [supportsz]
        :param x_qry:   [querysz, c_, f, t]
        :param y_qry:   [querysz]
        :return:
        """
        loss = 0
        pred = None
        for idx in range(0, ultra_list.size(0), self.batch_size):
            ultra = ultra_list[idx:idx+self.batch_size, ]
            voice = voice_list[idx:idx+self.batch_size, ]
            label = y_list[idx:idx+self.batch_size, ]
            data = torch.cat([ultra, voice], dim=1)
            out = model(data, vars)
            if pred==None:
                pred = out
            else:
                pred = torch.cat([pred, out], dim=0)
            loss += F.binary_cross_entropy(out, label, reduction='sum')

        return pred, loss/ultra_list.size(0)

    def forward(self, support_ultra, support_voice, support_y, query_ultra, query_voice, query_y):
        """

        :param x_spt:   [b, supportsz, c_, f, t]
        :param y_spt:   [b, supportsz]
        :param x_qry:   [b, querysz, c_, f, t]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, supportsz, c_, f, t = support_ultra.size()
        querysz = query_ultra.size(1)

        losses_q = 0#[0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]
        self.net.train()

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            # logits = self.net(x_spt[i], vars=None, bn_training=True)
            net = L2L.clone_module(self.net)
            # opt = optim.Adam(net.parameters(), lr=self.update_lr)
            _, loss = self.compute_loss(net, support_ultra[i], support_voice[i], support_y[i], vars=None)
            grad = torch.autograd.grad(loss, net.parameters())

            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))
            
            
     
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [supportsz, nway]
                # logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                pred_q, loss_q = self.compute_loss(net, query_ultra[i], query_voice[i], query_y[i], vars=self.net.parameters())
                # losses_q[0] += loss_q

                pred_q[pred_q>self.threshold] = 1
                pred_q[pred_q<=self.threshold] = 0
                correct = torch.eq(pred_q, query_y[i]).sum().item()
                corrects[0] = corrects[0] + correct

          
            # this is the loss and accuracy after the first update
            L2L.update_module(net, updates=fast_weights)
         
            with torch.no_grad():
                # [supportsz, nway]
                # logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                pred_q, loss_q = self.compute_loss(net, query_ultra[i], query_voice[i], query_y[i], vars=fast_weights)
                # losses_q[1] += loss_q
                # [supportsz]
                pred_q[pred_q>self.threshold] = 1
                pred_q[pred_q<=self.threshold] = 0
                correct = torch.eq(pred_q, query_y[i]).sum().item()
                corrects[1] = corrects[1] + correct
     
            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                # logits = self.net(x_spt[i], fast_weights, bn_training=True)
                _, loss = self.compute_loss(net, support_ultra[i], support_voice[i], support_y[i], vars=fast_weights)
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                L2L.update_module(net, updates=fast_weights)

                # logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                pred_q, loss_q = self.compute_loss(net, query_ultra[i], query_voice[i], query_y[i], vars=fast_weights)

                if k == self.update_step-1:
                    losses_q += loss_q
                
                

                with torch.no_grad():
                    pred_q[pred_q>self.threshold] = 1
                    pred_q[pred_q<=self.threshold] = 0
                    correct = torch.eq(pred_q, query_y[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct
            

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        # print('meta update')
        # for p in self.net.parameters()[:5]:
        # 	print(torch.norm(p).item())
        
        # for p in self.net.parameters():
        #     p.grad.data.mul_(1.0 / task_num)
   
        self.meta_optim.step()
   
        accs = np.array(corrects) / (querysz * task_num)

        return accs, loss_q.item()


    def finetunning(self, support_ultra, support_voice, support_y, query_ultra, query_voice, query_y, save=False, save_path=None):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(support_ultra.shape) == 4

        querysz = query_ultra.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]


        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = L2L.clone_module(self.net)
        # net.train()
        # 1. run the i-th task and compute loss for k=0
        # logits = net(x_spt)
        _, loss = self.compute_loss(net, support_ultra, support_voice, support_y)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            # logits_q = net(x_qry, net.parameters(), bn_training=True)
            pred_q, loss_q = self.compute_loss(net, query_ultra, query_voice, query_y, vars=self.net.parameters())
            # [setsz]
            pred_q[pred_q>self.threshold] = 1
            pred_q[pred_q<=self.threshold] = 0
            # scalar
            correct = torch.eq(pred_q, query_y).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        L2L.update_module(net, updates=fast_weights)
        with torch.no_grad():
            # [setsz, nway]
            # logits_q = net(x_qry, fast_weights, bn_training=True)
            pred_q, loss_q = self.compute_loss(net, query_ultra, query_voice, query_y, vars=fast_weights)
            # [setsz]
            pred_q[pred_q>self.threshold] = 1
            pred_q[pred_q<=self.threshold] = 0
            # scalar
            correct = torch.eq(pred_q, query_y).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            # logits = net(x_spt, fast_weights, bn_training=True)
            # net.train()
            _, loss = self.compute_loss(net, support_ultra, support_voice, support_y, vars=fast_weights)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
      
            L2L.update_module(net, updates=fast_weights)
            # logits_q = net(x_qry, fast_weights, bn_training=True)
     
            # loss_q will be overwritten and just keep the loss_q on last update step.
            # net.eval()
            pred_q, loss_q = self.compute_loss(net, query_ultra, query_voice, query_y, vars=fast_weights)

            with torch.no_grad():
                pred_q[pred_q>self.threshold] = 1
                pred_q[pred_q<=self.threshold] = 0  
                correct = torch.eq(pred_q, query_y).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct
        if save:
            torch.save({
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': self.meta_optim.state_dict(),  
            }, save_path)

        del net

        accs = np.array(corrects) / querysz

        return accs, loss_q.item()

    def save_model(self, save_path):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.meta_optim.state_dict(),  
            }, save_path)

    def load_model(self, save_path):
        ckpt = torch.load(save_path)
        self.net.load_state_dict(ckpt['model_state_dict'])
        self.meta_optim.load_state_dict(ckpt['optimizer_state_dict'])

def main():
    pass


if __name__ == '__main__':
    main()
