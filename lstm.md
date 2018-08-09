# LSTM的梯度消失和反向传播

## 权重值的梯度
首先，我们先看下LSTM的所有公式：

- $ f_t = \sigma(W_{xf}x_t + W_{hf}h_{t-1} + b_f)$
- $ i_t = \sigma(W_{xi}x_t + W_{hi}h_{t-1} + b_i)$
- $ g_t = tanh(W_{xg}x_t + W_{hg}h_{t-1} + b_g)$
- $ c_t = f_t \circ c_{t-1} + i_t \circ g_t $
- $ o_t = \sigma(W_{xo}x_t + W_{ho}h_{t-1} + b_o)$
- $ h_t = o_t \circ tanh(c_t)$
- $ z_t = W_{hz}h_t + b_z$

**从一个简单的loss function出发，我们使用Cross Entropy Loss.**

- $CE = -\sum_i y_i log \sigma(z_t)$

**因此，我们就可以通过CE loss来计算所有权重的权值。**
![Alt text](./1533748865175.png)
- 首先，可以得到以下的结果
	- $ dz_t = y_t - z_t$
	- $dW_{hz} = h_t \cdot dz_t$
	- $ dh_T = W_{hz} dz_T$ (考虑序列最后一个时间点T，其他时间t的结果略有不同)
- 根据这些结果，我们就可以继续往下推了
	- $ dc_t = (1- tanh(c_t)^2) \cdot dh_t $
	- $ do_t = tanh(c_t) \cdot dh_t$
	- $ dg_t = i_t \cdot dc_t$
	- $ di_t = g_t \cdot dc_t$
	- $ df_t = c_{t-1} \cdot dc_t$
	- $ dc_{t-1} += f_t \cdot dc_t$

- 根据上面的中间结果，我们可以看到，在BPTT中，所有的error flow都是通过$c_t$传递的！
- **而权重值的计算也很类似，不过需要考虑到$W_{xo}, W_{xi}, W_{xf}, W_{xg}$在全局都是共享的，同步更新，所以需要叠加到t时刻。**
	- $dW_{xo} = \sum_t o_t  (1-o_t) x_tdo_t$
	- $dW_{xf} = \sum_t f_t  (1-f_t) x_tdf_t$
	- $dW_{xi} = \sum_t i_t  (1-i_t) x_tdi_t$
	- $dW_{xg} = \sum_t (1-g_t^2) x_tdg_t$

- 当然，关于h的权重也都是一样，就不一一细推了。
- **最后，这里考虑的都是$dh_t$已知的情况，那如何来推$dh_t$呢？**
	- (分两种情况，根据递推和根据此时的loss)
		- $dh_{t-1} = o_t (1-o_t)W_{ho}do_t + f_t(1-f_t)W_{hf}df_t + i_t(1-i_t)W_{hi}di_t + (1-g_t^2)W_{hg}dg_t$
		- $ dh_{t-1} = dh_{t-1} + W_{hz}dz_t $

## BP for $c_t$

- 在T步，我们可以得到$c_t$对于T步的loss的梯度如下：
	- $\frac{\partial L_T}{\partial c_T} = \frac{\partial L_T}{\partial h_T} \frac{\partial h_T}{\partial c_T} $
- 同样的，我们可以得到$c_{T-1}$对于T-1步的loss：
	- $\frac{\partial L_{T-1}}{\partial c_{T-1}} = \frac{\partial L_{T-1}}{\partial h_{T-1}} \frac{\partial h_{T-1}}{\partial c_{T-1}} $
- 而此时，$c_{T-1}$也有从T步BP传回的loss，因此它的结果是两者的和：
	- $\frac{\partial L_{T-1}}{\partial c_{T-1}} =  \frac{\partial L_{T-1}}{\partial h_{T-1}} \frac{\partial h_{T-1}}{\partial c_{T-1}}  + \frac{\partial L_T}{\partial h_T} \frac{\partial h_T}{\partial c_T}  \frac{\partial c_T}{\partial c_{T-1}}$
- 因此，我们根据上面的公示重写上式得：
	- $ dc_{T-1} = dc_{T-1} + f_T \circ dc_{T} $

## 结论
- 因此，可以得到结论，如果是正常的序列预测问题，每一步都存在loss，那是不会出现梯度消失的情况的。
- 当问题的loss只定义在T点时，
	- $dc_{t-1} = f_t \circ dc_{t} = dc_T\prod_t^T f_t $
	- 因此如果每一步的$f_t=1$则不会存在梯度消失的问题，而如果传播中一些$f_t \approx 0$时，本来就根据$f_t$的定义为不需要继续记忆，所以梯度消失也可以接受。
