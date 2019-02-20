import torch
from torch import nn
import copy
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

bidirectional = False
T = 35
N = 10
L = 4
D = 2 if bidirectional else 1
I = 200
H = 200

def f(t1, t2, msg):
    def to_numpy(t):
        return t.detach().numpy() if t.requires_grad else t.numpy()

    s1 = t1.sum().item()
    s2 = t2.sum().item()
    re = np.allclose(to_numpy(t1), to_numpy(t2), atol=1e-06)
    print("%s:\t%s, %.6f, \t%.6f" % (msg, 'PASS' if re else 'FAIL', s1, s2))


def test_lstm_forward():
    print("\n##### lstm forward #####")
    input = torch.randn(T, N, I)
    h0 = torch.randn(L*D, N, H)
    c0 = torch.randn(L*D, N, H)

    torch._C._set_mkldnn_enabled(True)
    rnn1 = nn.LSTM(I, H, L, bidirectional=bidirectional)
    rnn1.eval()
    output1, hn1 = rnn1(input, (h0, c0))
    hy1, cy1 = hn1

    torch._C._set_mkldnn_enabled(False)
    rnn2 = copy.deepcopy(rnn1)
    output2, hn2 = rnn2(input, (h0, c0))
    hy2, cy2 = hn2

    f(output1, output2, 'output  ')
    f(hy1, hy2, 'hidden  ')
    f(cy1, cy2, 'cell    ')

def test_gru_forward():
    print("\n##### gru forward ######")
    input = torch.randn(T, N, I)
    h0 = torch.randn(L*D, N, H)

    torch._C._set_mkldnn_enabled(True)
    rnn1 = nn.GRU(I, H, L, bidirectional=bidirectional)
    rnn1.eval()
    output1, hn1 = rnn1(input, h0)

    torch._C._set_mkldnn_enabled(False)
    rnn2 = copy.deepcopy(rnn1)
    output2, hn2 = rnn2(input, h0)

    f(output1, output2, 'output  ')
    f(hn1, hn2, 'hidden  ')

def test_gru_backward(grad_x=False, grad_hx=False):
    print("\n##### gru backward: grad_x %s grad_hx %s #####" % ('True' if grad_x else 'False',
                                                                'True' if grad_hx else 'False'))
    x1 = torch.randn(T, N, I)
    x2 = x1.clone()
    hx1 = torch.randn(L*D, N, H)
    hx2 = hx1.clone()

    if grad_x:
        x1.requires_grad_(True)
        x2.requires_grad_(True)

    if grad_hx:
        hx1.requires_grad_(True)
        hx2.requires_grad_(True)

    torch._C._set_mkldnn_enabled(True)
    rnn1 = nn.GRU(I, H, L, bidirectional=bidirectional)
    output1, hy1 = rnn1(x1, hx1)
    output1.mean().backward(retain_graph=True)

    torch._C._set_mkldnn_enabled(False)
    rnn2 = copy.deepcopy(rnn1)
    output2, hy2 = rnn2(x2, hx2)
    output2.mean().backward(retain_graph=True)

    for l in range(len(rnn1.all_weights)):
        layer = l / D
        direction = l % D
        l1 = rnn1.all_weights[l]
        l2 = rnn2.all_weights[l]
        f(l1[0].grad.data, l2[0].grad.data, 'L %d D %d w1' % (layer, direction))
        f(l1[1].grad.data, l2[1].grad.data, 'L %d D %d w2' % (layer, direction))
        f(l1[2].grad.data, l2[2].grad.data, 'L %d D %d b1' % (layer, direction))
        f(l1[3].grad.data, l2[3].grad.data, 'L %d D %d b2' % (layer, direction))

    if grad_x:
        f(x1.grad.data, x2.grad.data, 'grad input')

    if grad_hx:
        f(hx1.grad.data, hx2.grad.data, 'grad hx')

def test_lstm_backward(grad_x=False, grad_hx=False, grad_cx=False):
    print("\n##### lstm backward: grad_x %s grad_hx %s  grad_cx %s #####" % ('True' if grad_x else 'False',
                                          'True' if grad_hx else 'False', 'True' if grad_cx else 'False'))

    x1 = torch.randn(T, N, I)
    x2 = x1.clone()

    hx1 = torch.randn(L*D, N, H)
    hx2 = hx1.clone()
    cx1 = torch.randn(L*D, N, H)
    cx2 = cx1.clone()

    if grad_x:
        x1.requires_grad_(True)
        x2.requires_grad_(True)

    if grad_hx:
        hx1.requires_grad_(True)
        hx2.requires_grad_(True)

    if grad_cx:
        cx1.requires_grad_(True)
        cx2.requires_grad_(True)

    torch._C._set_mkldnn_enabled(True)
    rnn1 = nn.LSTM(I, H, L, bidirectional=bidirectional)
    output1, hn1 = rnn1(x1, (hx1, cx1))
    hy1, cy1 = hn1
    output1.mean().backward(retain_graph=True)

    torch._C._set_mkldnn_enabled(False)
    rnn2 = copy.deepcopy(rnn1)
    output2, hn2 = rnn2(x2, (hx2, cx2))
    hy2, cy2 = hn2
    output2.mean().backward(retain_graph=True)

    for l in range(len(rnn1.all_weights)):
        layer = l / D
        direction = l % D
        l1 = rnn1.all_weights[l]
        l2 = rnn2.all_weights[l]
        f(l1[0].grad.data, l2[0].grad.data, 'L %d D %d w1' % (layer, direction))
        f(l1[1].grad.data, l2[1].grad.data, 'L %d D %d w2' % (layer, direction))
        f(l1[2].grad.data, l2[2].grad.data, 'L %d D %d b1' % (layer, direction))
        f(l1[3].grad.data, l2[3].grad.data, 'L %d D %d b2' % (layer, direction))

    if grad_x:
        f(x1.grad.data, x2.grad.data, 'grad input')

    if grad_hx:
        f(hx1.grad.data, hx2.grad.data, 'grad hx')

    if grad_cx:
        f(cx1.grad.data, cx2.grad.data, 'grad cx')


test_lstm_forward()
test_gru_forward()
#test_gru_backward()
test_gru_backward(grad_x=True, grad_hx=True)
#test_lstm_backward()
test_lstm_backward(grad_x=True, grad_hx=True, grad_cx=True)
