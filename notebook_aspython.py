# %% [markdown]
# <a href="https://colab.research.google.com/github/yifanlu0227/MIT-6.5940/blob/main/Lab4.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # RC: systems for ML

# %% [markdown]
# ## Introduction
# 
# This colab notebook provides code and a framework for Lab 4: LLM Quantization. You will learn how to quantize a large language model that can run efficiently. We will implement AWQ (activation aware weight only quantization) for 4 bit weight-only quantization.
# 
# Running large language models (LLMs) on the edge is of great importance, which not only enhances user experience but also addresses privacy concerns, as sensitive data remains localized and reduces the risk of potential breaches.
# 
# However, deploying LLMs on the edge presents significant challenges. Edge devices operate under tight power constraints, setting them apart from workstations or cloud servers. This translates to restricted memory bandwidth and limited peak computation throughput on the edge. For instance, the NVIDIA Jetson Orin Nano, with its 8GB DRAM, cannot accommodate even the most compact LLaMA-2 model in half precision. Thankfully, AWQ presents a push-the-button solution for weight quantization, empowering LLM inference on edge devices with constrained memory.
# 
# Furthermore, by using the AWQ 4-bit weight-only quantization algorithm, combined with an efficient 4-bit kernel, we can achieve the following acceleration on the RTX 4090. In the next lab section, we will also use TinyChatEnigne to achieve actual performance acceleration.

# %% [markdown]
# ### Demo on an RTX 4090:
# 

# %% [markdown]

# %% [markdown]
# ### Demo on an Apple MacBook Air (M1, 2020):
# 

# %% [markdown]
# ![demo.gif](https://github.com/mit-han-lab/TinyChatEngine/blob/main/assets/figures/chat_demo_m1.gif?raw=true)

# %% [markdown]
# # AWQ (activation aware weight only quantization)

# %% [markdown]

# %% [markdown]
# Large language models (LLMs) have shown excellent performance on various tasks, but the astronomical model size raises the hardware barrier for serving (memory size) and slows down token generation (memory bandwidth). LLM sizes and computation are increasing exponentially, while memory bandwidth is increasing slowly. This gap is a major bottleneck for LLMs. In this lab, we will explore the use of an novel quantization algorithm (AWQ) to reduce memory footprint of LLMs and achieve accelerations for inference.

# %% [markdown]
# In previous courses, we have learned the basic methods of quantization.
# There are two types of quantization:
# 
# - Quantize both weight and activation
#     - Better for computation-bounded scenarios: context stage, large batch inference
#     - For example, SmoothQuant: W8A8 quantization
# - Weight-only quantization
#     - Better for memory-bounded scenarios: decoding stage, single batch inference
#     - For example, AWQ that will be introduced in this lab: W4A16 quantization

# %% [markdown]
# For the LLaMA-65B model, in the decoding stage of single batch inference, we need to perform GEMV $[1, 8192] \times [8192, 8192]$. Taking the NVIDIA A100 80G as an example, its half-precision (FP16) performance is 312TFLOPS, and the memory bandwidth is about 2000GB/s. Therefore, its computation intensity is:
# 
# $$
# \frac{\text{FLOP}}{\text{Byte}} = \frac{2\times 8192^2}{8192^2} << \frac{3.12\times 10^{11}}{2\times 10^9}
# $$
# 
# This is very memory-bounded (~$10^2$ gap), which is why we need low-bit weight quantization.

# %% [markdown]
# ## Setup

# %%
print('Installing packages...')
!pip install torch transformers==4.31.0 accelerate==0.21.0 sentencepiece==0.1.99 tokenizers==0.13.3 datasets==2.14.4 tqdm zstandard

# %%
import tqdm
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from functools import partial
import gc

# %% [markdown]
# Here we use wikitext-2 dataset for evaluation. The dataset is automatically downloaded by the code.

# %%
def evaluate(model, tokenizer):
    testenc = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    testenc = tokenizer("\n\n".join(testenc['text']), return_tensors='pt')

    testenc = testenc.input_ids.to(model.device)
    nsamples = 40
    model = model.eval()

    nlls = []
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * 2048):((i + 1) * 2048)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * 2048):((i + 1) * 2048)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * 2048
        nlls.append(neg_log_likelihood)

    return torch.exp(torch.stack(nlls).sum() / (nsamples * 2048))


# %% [markdown]
# The following code is used to calculate the model size.

# %%
def get_model_size(model: nn.Module, data_width=16, group_size=-1):

    if group_size != -1:
        data_width += (16 + 4) / group_size

    num_elements = 0
    for param in model.parameters():
        num_elements += param.numel()
    return num_elements * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

# %% [markdown]
# Let's first evaluate the perplexity and model size of the FP32 Model.

# %%
model_path = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")


# %%
# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=32, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# Uniform quantization is to map real values in the range $[\beta, \alpha]$ to lie within $[0, 2^{b} - 1]$.
# 
# Notation:
# 
# - Quantized Weight: $w_q$
# 
# - Scale factor: $s_q$
# 
# - Zero Point: $z$
# \begin{equation}
# s_q = \frac{\alpha - \beta}{2^{b} - 1} \tag{1},
# \end{equation}
# \begin{equation}
# z = -\text{Round}(\beta * scale) \tag{2}
# \end{equation}
# \begin{equation}
# w_q = \text{Clamp}(\text{Round}(\frac{w}{s_q}) + z) \tag{3},
# \end{equation}
# 
# 

# %% [markdown]
# ### pseudo quantization
# The following code is for pseudo quantization.
# 
# Pseudo Quantization is used to simulate the effects of quantization on a model  without actually quantizing the model's weights. (i.e. rounding to the nearest quantized value and then **dequantizing back to a float**.)

# %%
# core quantization method (simulated quantization)
def pseudo_quantize_tensor(w, n_bit=4, q_group_size=-1):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate the maximum (\alpha) and minimum values (\beta) in the tensor.
    max_val = w.amax(dim=1, keepdim=True)
    assert max_val.dim() == 2 and max_val.size(0) == w.size(0) and max_val.size(1) == 1
    min_val = w.amin(dim=1, keepdim=True)
    assert min_val.dim() == 2 and min_val.size(0) == w.size(0) and min_val.size(1) == 1

    # Calculate the scale factor and zero point.  (Formula 1 & 2)
    max_int = 2 ** n_bit - 1
    scales = (max_val - min_val).clamp(min=1e-5) / max_int
    assert scales.shape == max_val.shape
    zeros = (-torch.round(min_val / scales)).clamp_(0, max_int)
    assert scales.shape == min_val.shape

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    # Quantize W: Map values in the range [\beta, \alpha] to lie within [0, 2^b - 1] (Formula 3)
    w = torch.clamp(torch.round(w / scales) + zeros, 0, max_int)
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    # Dequantize W (pseudo quantization, the inverse transformation of Formula 3)
    w = (w - zeros) * scales
    assert w.dim() == 2 and w.size(0) == scales.size(0) and w.size(1) == q_group_size

    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)
    return w

@torch.no_grad()
def pseudo_quantize_model_weight(
    model, w_bit, q_group_size,
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

# %% [markdown]
# Let's evaluate the perplexity and model size of the quantized 3-bit Model.

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight(model, w_bit=3, q_group_size=128)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# We can see that the model size has decreased, but the perplexity has significantly increased.

# %% [markdown]
# There is a observation in LLM activations that **outliers appear in a small fraction of the channels**. If one channel has an outlier, it **persistently appears in all tokens**. The variance amongst the channels for a given token is large (the activations in some channels are very large, but most are small), but the variance between the magnitudes of a given channel across tokens is small (outlier channels are consistently large).
# 
# According to the observation of AWQ, weight channels corresponding to activation outliers are more salient, and preserving those salient weights can lead to a significant performance improvement. Next, let's try to find the salient weights and retain them as original values to observe the change in perplexity.
# 
# The following code is used to load the calibration dataset, so as to obtain activation outliers to identify salient weights.

# %%
def get_calib_dataset(tokenizer=None, n_samples=256, block_size=512):
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break

    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [cat_samples[:, i*block_size:(i+1)*block_size] for i in range(n_split)]

@torch.no_grad()
def get_calib_feat(model, tokenizer):
    input_dict = dict()
    def stat_input_max_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        x_max = x.view(-1, x.shape[-1]).abs().mean(dim=0).cpu().detach()
        if name not in input_dict:
            input_dict[name] = [x_max]
        else:
            input_dict[name] += [x_max]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    partial(stat_input_max_hook, name=name)))

    print("Collecting activation scales...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = get_calib_dataset(tokenizer)
    pbar = tqdm.tqdm(samples)
    for input_ids in pbar:
        input_ids = input_ids.to(device)
        model(input_ids)

    for hook in hooks:
        hook.remove()
    return input_dict

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
input_feat = get_calib_feat(model, tokenizer)

# %%
print(type(input_feat['model.decoder.layers.0.self_attn.q_proj']))
print(len(input_feat['model.decoder.layers.0.self_attn.q_proj']))
print(input_feat['model.decoder.layers.0.self_attn.q_proj'][0].shape)
print(sum(input_feat['model.decoder.layers.0.self_attn.q_proj']).shape)

# %% [markdown]
# ### Question 1 (50 pts)
# #### Question 1.1 (20 pts)
# Next, please add codes before and after the quantization to protect 1% of the salient weight channels (1% channels with highest importance), ensuring that their values remain unchanged after quantization. (**The desired perplexity is 17.15**)

# %%
@torch.no_grad()
def pseudo_quantize_model_salient_weight_fp16(
    model, w_bit, q_group_size, input_feat
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # Step 1: Find 1% of the salient weight channels according to importance (hint: use torch.topk())
            outlier_indices = torch.topk(importance, int(len(importance) * 0.01))[1]
            assert outlier_indices.dim() == 1

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the salient weight channels
            outlier = m.weight.data[:, outlier_indices].clone()

            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Restore the 1% salient weight channels to their original FP16 values
            m.weight.data[:, outlier_indices] = outlier

            ############### YOUR CODE ENDS HERE #################

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_salient_weight_fp16(model, w_bit=3, q_group_size=128, input_feat=input_feat)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# #### Question 1.2 (15 pts)
# Let's conduct an ablation experiment: randomly protect 1% of the weight channels, ensuring that their values remain unchanged after quantization, and then observe the perplexity. (**The desired perplexity is over 100**)
# 
# 
# 

# %%
@torch.no_grad()
def pseudo_quantize_model_random_weight_fp16(
    model, w_bit, q_group_size, input_feat
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # Step 1: Randomly choose 1% of the weight channels
            outlier_mask = torch.randint(0, len(importance), (int(len(importance)*0.01), ))
            assert outlier_mask.dim() == 1

            ############### YOUR CODE ENDS HERE #################

            # Back up the values of the selected weight channels
            outlier = m.weight.data[:, outlier_mask].clone()

            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Restore the 1% selected weight channels to their original FP16 values
            m.weight.data[:, outlier_mask] = outlier

            ############### YOUR CODE ENDS HERE #################

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_random_weight_fp16(model, w_bit=3, q_group_size=128, input_feat=input_feat)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# #### Question 1.3 (15 pts)
# Please provide a possible explanation for why the salient weight channels are so important.
# 
# #### Answser 1.3
# ############### YOUR ANSWER STARTS HERE #################
# 
# Salient weight channels correspond to those **outlier channels in activations** (with very large values). If we quantize them, there will be a significant drop in numerical accuracy.
# 
# ############### YOUR ANSWER ENDS HERE #################

# %% [markdown]
# ### Question 2 (50 pts)

# %% [markdown]
# Despite keeping 0.1% of weights in FP16 can improve the quantized performance
# without a noticeable increase in model size (measured in total bits), such a mixed-precision data type will make the system implementation difficult. We need to come up with a method to protect the important weights without actually keeping them as FP16.

# %% [markdown]
# According to the methodology of AWQ, simply scaling up the salient weight channels can protect them. The principle is as follows:
# 
# - Consider a linear layer channel $\mathbf{y} = \mathbf{w}x$ (from $\mathbf{W}x$). We care about the quantization error from $Q(\mathbf{w})x$.
# 
# - $Err(Q(\mathbf{w}) x) = Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x$, $Δ = \frac{\max(|w|)}{2^{N - 1}}$.
# - The scaled version is $Err(Q(\mathbf{w} \cdot s)(\frac{x}{s})) = Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x\cdot \mathbf{\frac{1}{s}}$.
# - The $RoundErr$ is always ~0.25 (average from 0-0.5).
# - When the group size is relatively large (e.g., 128), scaling up one channel usually does not increase the maximum value in a group (i.e. $Δ$ remains unchanged).
# - Thus, $Err(Q(\mathbf{w} \cdot s)(\frac{x}{s})) = Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x\cdot \mathbf{\frac{1}{s}}$ < $Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x = Err(Q(\mathbf{w}) x)$.

# %% [markdown]
# Taking the following figure as an example, if we assume 3-bit int quantization, then the quantization error caused by the value in the last column of the second row of $W(+1.4)$ should be $Err(Q(\mathbf{w}) x) = Δ\cdot RoundErr(\frac{\mathbf{w}}{Δ})\cdot x$ = $\frac{4}{2^{3 - 1}} * |1.4 - 1.0| * (2 + 2 + 2) = 2.4$.
# 
# If the second channel is scaled up by a factor of $2$, the resulting quantization error would reduce to $\frac{4}{2^{3 - 1}} * |2.8 - 3.0| * (2/2 + 2/2 + 2/2) = 0.6$.

# %% [markdown]

# %% [markdown]
# #### Question 2.1 (20 pts)
# Please write code to scale up the salient weight channels, then quantize it, and finally scale it back down, and observe the changes in perplexity. (**The desired perplexity is 18.93**)

# %%
@torch.no_grad()
def pseudo_quantize_model_weight_scaleup(
    model, w_bit, q_group_size, input_feat, scale_factor
):
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            importance = sum(input_feat[n]).float()

            ############### YOUR CODE STARTS HERE ###############

            # Step 1: Find 1% of the salient weight channels
            outlier_mask = torch.topk(importance, int(len(importance) * 0.01))[1]
            assert outlier_mask.dim() == 1

            ############### YOUR CODE ENDS HERE #################

            # To simulate applying the scale factor, we can simply multiply it before quantization, and then divide by the scale factor after quantization.
            # Scale up the values of the salient weight channels
            m.weight.data[:, outlier_mask] *= scale_factor

            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Scale back down the values of the salient weight channels
            m.weight.data[:, outlier_mask] /= scale_factor

            ############### YOUR CODE ENDS HERE #################

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight_scaleup(model, w_bit=3, q_group_size=128, input_feat=input_feat, scale_factor=2)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %%
for scale_factor in [1,2,3,4]:
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    pseudo_quantize_model_weight_scaleup(model, w_bit=3, q_group_size=128, input_feat=input_feat, scale_factor=scale_factor)

    # Evaluate the model
    model_perplexity = evaluate(model, tokenizer)
    model_size = get_model_size(model, data_width=3, group_size=128)
    print(f"{scale_factor=}")
    print(f"\nmodel perplexity: {model_perplexity:.2f}")
    print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# #### Question 2.2 (15 pts)
# Please try different scale factors (e.g. 1, 2, 3, and 4) in the code and observe the changes in perplexity.
# 
# Did you observe the perplexity first decreasing and then increasing? Please explain why this would happen based on the principle mentioned above.
# 
# #### Answer 2.2
# ############### YOUR ANSWER STARTS HERE #################
# 
# Scaling up at a large factor may increase the maximum value in the group (i.e.  Δ will increase). It will affect other channel's quantization.
# 
# ############### YOUR ANSWER ENDS HERE #################

# %% [markdown]
# ### Question 2.3 (15 pts)
# Due to the instability of fine-tuning, it would be a better choice to find the optimal $s$ within a predefined search space. We can find the optimal scale in the search space to protect the salient weights while also considering other values. In practice, it can be observed that considering just the activations is sufficient to yield good results. Please add the code for search and run it to observe the perplexity. (**The desired perplexity is 17.92**)

# %% [markdown]
# $$
# 𝐋(\mathbf{s})=\lVert Q(\mathbf{W}\cdot \mathbf{s})  (\mathbf{s^{-1}} \cdot \mathbf{X}) - \mathbf{W}\mathbf{X}  \rVert,  \quad\mathbf{s}= \mathbf{s_X}^{\alpha}
# $$
# $$
# \mathbf{s}^* = \text{argmin}_{\mathbf{s}} 𝐋(\mathbf{s}),\quad \alpha^*=\text{argmin}_{\alpha} 𝐋(\mathbf{s_X}^{\alpha})
# $$

# %%
@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)

    ln.weight.div_(scales)
    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    scales = scales.to(fc1.weight.device)

    # fc1.weight.div_(scales.view(-1, 1))
    fc1.weight[-scales.size(0):].div_(scales.view(-1, 1))
    if fc1.bias is not None:
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0

@torch.no_grad()
def auto_scale_block(module, name, w_bit,
                     q_group_size,
                     input_feat):

    # find the best scale ratio
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):

        x = x.to(next(block.parameters()).device)
        with torch.no_grad():
            org_out = block(x, **kwargs)
            if isinstance(org_out, tuple):
                org_out = org_out[0]

        s_x = x.view(-1, x.shape[-1]).abs().mean(0)

        ############### YOUR CODE STARTS HERE ###############

        # Step 1: Initialize the best_error, best_ratio and best_scales
        best_error = torch.inf
        best_ratio = -1
        best_scales = 0

        ############### YOUR CODE ENDS HERE #################

        n_grid = 20
        history = []

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}
        for ratio in range(n_grid):
            # ratio is the \alpha in the formula
            ratio = ratio * 1 / n_grid

            ############### YOUR CODE STARTS HERE ###############

            # Step 2: Calculate the scales by the formula: scales = s_x^ratio
            scales = torch.clamp(s_x, 1e-5) ** ratio # must clip the s_x, otherwise will get nan later

            assert scales.shape == s_x.shape

            ############### YOUR CODE ENDS HERE #################

            scales = scales / (scales.max() * scales.min()).sqrt().view(1, -1)

            for fc in linears2scale:

                scales = scales.to(fc.weight.device)

                # Scale up the values of the weight channels
                fc.weight.mul_(scales)

                fc.weight.data = pseudo_quantize_tensor(fc.weight.data, w_bit, q_group_size)

                ############### YOUR CODE STARTS HERE ###############

                # Step 3: Scale back down the values of the weight channels
                fc.weight.data /= scales

                ############### YOUR CODE ENDS HERE #################

            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = (org_out - out).float().pow(2).mean().item()  # float prevents overflow
            history.append(loss)
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            block.load_state_dict(org_sd)

        if best_ratio == -1:
            print(history)
            raise Exception

        best_scales = best_scales.view(-1)

        assert torch.isnan(best_scales).sum() == 0, best_scales
        return best_scales.detach()

    # attention input
    inp = input_feat[name + '.self_attn.out_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0).unsqueeze(0)
    qkv = [module.self_attn.q_proj, module.self_attn.k_proj, module.self_attn.v_proj]
    final_scales = _search_module_scale(module.self_attn, qkv, inp)
    scale_ln_fcs(module.self_attn_layer_norm, qkv, final_scales)

    # attn out
    inp = input_feat[name + '.self_attn.out_proj']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.self_attn.out_proj, [module.self_attn.out_proj], inp)
    scale_fc_fc(module.self_attn.v_proj, module.self_attn.out_proj, final_scales)

    # fc1
    inp = input_feat[name + '.fc1']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc1, [module.fc1], inp)
    scale_ln_fcs(module.final_layer_norm, module.fc1, final_scales)

    # fc2
    inp = input_feat[name + '.fc2']
    inp = torch.cat([x.unsqueeze(0) for x in inp], dim=0)
    final_scales = _search_module_scale(module.fc2, [module.fc2], inp)
    scale_fc_fc(module.fc1, module.fc2, final_scales)

@torch.no_grad()
def pseudo_quantize_model_weight_auto_scale(
    model, w_bit, q_group_size, input_feat
):
    from transformers.models.opt.modeling_opt import OPTDecoderLayer

    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            auto_scale_block(module, name, w_bit, q_group_size, input_feat)

    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, q_group_size=q_group_size)

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pseudo_quantize_model_weight_auto_scale(model, w_bit=3, q_group_size=128, input_feat=input_feat)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# # Power of Two addition

# %%


# %% [markdown]
# # Task
# Implement Power of Two (POT) and Additive Power of Two (APOT) post-training quantization strategies based on the paper "/content/potptq paper.pdf". Add new code blocks to the notebook for the implementation, calibration, and evaluation (model size, perplexity, etc.) of these methods.

# %% [markdown]
# ## Implement pot quantization
# 
#  function to perform Power of Two (POT) quantization on model weights.
# 

# %%
def pot_quantize_tensor(w, n_bit=4, q_group_size=-1):
    """
    Power-of-Two quantization following the POT-PTQ paper.

    Key differences from uniform quantization:
    - Weights are represented as: w_q = scale * sign(w) * 2^E
    - Uses grid search for optimal scale initialization
    - Quantization levels are logarithmically spaced

    Args:
        w: Weight tensor to quantize
        n_bit: Number of bits for quantization
        q_group_size: Group size for quantization (-1 for per-channel)
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Maximum exponent for n-bit quantization
    q_max = 2**(n_bit - 1) - 1

    # Step 1: Initialize base scale (Eq. 10 in paper)
    max_val = w.abs().amax(dim=1, keepdim=True)
    s_0 = max_val / (2**q_max - 1)
    s_0 = torch.clamp(s_0, min=1e-5)  # Avoid division by zero

    # Step 2: Grid search for optimal scale multiplier (Algorithm 1)
    # Search range: b ∈ [0.01, 2.0] with step 0.01
    B = torch.arange(0.01, 2.01, 0.01, device=w.device)
    best_error = torch.full((w.size(0), 1), float('inf'), device=w.device)
    best_scale = s_0.clone()

    for b in B:
        # Candidate scale
        s_b = s_0 * b

        # Compute exponent: E = clamp(round(log2(|w| / s_b)), 0, q_max)
        # Using log2 for power-of-two quantization
        with torch.no_grad():
            ratio = torch.clamp(w.abs() / s_b, min=1e-10)
            E = torch.clamp(torch.round(torch.log2(ratio)), 0, q_max)

        # Reconstruct: w_q = s_b * sign(w) * 2^E
        w_q = s_b * torch.sign(w) * torch.pow(2.0, E)

        # Compute MSE (Eq. 8)
        error = ((w - w_q) ** 2).sum(dim=1, keepdim=True)

        # Update best scale
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, s_b, best_scale)

    # Final quantization with best scale
    with torch.no_grad():
        ratio = torch.clamp(w.abs() / best_scale, min=1e-10)
        E = torch.clamp(torch.round(torch.log2(ratio)), 0, q_max)

    # Dequantize: w = scale * sign(w) * 2^E
    w_quantized = best_scale * torch.sign(w) * torch.pow(2.0, E)

    assert torch.isnan(w_quantized).sum() == 0
    w_quantized = w_quantized.reshape(org_w_shape)

    return w_quantized

@torch.no_grad()
def pot_quantize_model_weight(model, w_bit, q_group_size):
    """
    Apply POT quantization to all linear layers in the model.

    Args:
        model: The model to quantize
        w_bit: Number of bits for weight quantization
        q_group_size: Group size for quantization
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = pot_quantize_tensor(
                module.weight.data,
                n_bit=w_bit,
                q_group_size=q_group_size
            )
    return model


# %% [markdown]
# ## Implement apot quantization
# 
# function to perform Additive Power of Two (APOT) quantization on model weights.
# 

# %%
def apot_quantize_tensor(w, n_bit=4, q_group_size=-1, n_addends=2):
    """
    Additive Power-of-Two quantization.

    APOT represents weights as a sum of power-of-two terms:
    w_q = scale * sum(sign_i * 2^E_i) for i=1 to n_addends

    This provides more flexibility than pure POT while maintaining
    hardware efficiency through shift-and-add operations.

    Args:
        w: Weight tensor to quantize
        n_bit: Number of bits for quantization
        q_group_size: Group size for quantization
        n_addends: Number of power-of-two terms to sum (default=2)
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # For APOT, we need to allocate bits across multiple addends
    # With n_addends=2 and n_bit=4, we could use 2 bits per addend
    bits_per_addend = max(1, n_bit // n_addends)
    q_max = 2**bits_per_addend - 1

    # Initialize scale similar to POT
    max_val = w.abs().amax(dim=1, keepdim=True)
    scale = max_val / (2**q_max * n_addends)
    scale = torch.clamp(scale, min=1e-5)

    # Iterative approximation: greedily find power-of-two addends
    w_residual = w.clone()
    w_quantized = torch.zeros_like(w)

    for i in range(n_addends):
        # Find the dominant power-of-two component in residual
        with torch.no_grad():
            ratio = torch.clamp(w_residual.abs() / scale, min=1e-10)
            E = torch.clamp(torch.round(torch.log2(ratio)), 0, q_max)

        # Compute this addend
        addend = scale * torch.sign(w_residual) * torch.pow(2.0, E)

        # Accumulate and update residual
        w_quantized += addend
        w_residual -= addend

    assert torch.isnan(w_quantized).sum() == 0
    w_quantized = w_quantized.reshape(org_w_shape)

    return w_quantized

@torch.no_grad()
def apot_quantize_model_weight(model, w_bit, q_group_size, n_addends=2):
    """
    Apply APOT quantization to all linear layers in the model.

    Args:
        model: The model to quantize
        w_bit: Number of bits for weight quantization
        q_group_size: Group size for quantization
        n_addends: Number of power-of-two addends
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = apot_quantize_tensor(
                module.weight.data,
                n_bit=w_bit,
                q_group_size=q_group_size,
                n_addends=n_addends
            )
    return model



# %% [markdown]
# ## Integrate pot evaluation
# 
# ### Subtask:
# Add code to evaluate the model after applying POT quantization, including calculating perplexity and model size.
# 

# %% [markdown]
# **Reasoning**:
# Delete the current model, clear the CUDA cache, load the pre-trained model, apply POT quantization, evaluate the quantized model, calculate the model size, and print the results.
# 
# 

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pot_quantize_model_weight(model, w_bit=4, q_group_size=128)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# ## apot evaluation
# 

# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
apot_quantize_model_weight(model, w_bit=2, q_group_size=128, n_addends=4)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# ### Power of Two (POT) Quantization
# 
# Power of Two (POT) quantization is a method where the scaling factor for quantization is restricted to be a power of two ($2^k$). This simplifies the dequantization process as it can be implemented with efficient bit shifts instead of floating-point multiplications.
# 
# The `pot_quantize_tensor` function implements this by:
# 1. For each group of weights (or the entire tensor if `q_group_size` is -1), it finds the maximum absolute value.
# 2. It then calculates the scale as the smallest power of two that is greater than or equal to this maximum absolute value divided by the maximum representable integer value for the given number of bits (`2^(n_bit-1) - 1`). This ensures that the quantized values fall within the representable range.
# 3. The weights are then quantized by dividing by this power-of-two scale and rounding to the nearest integer, clamping the results to the valid integer range.
# 4. Finally, the weights are dequantized by multiplying by the same power-of-two scale.
# 
# This method aims to maintain quantization efficiency while enabling faster inference due to the simplified dequantization operation.

# %% [markdown]
# ### Additive Power of Two (APOT) Quantization
# 
# Additive Power of Two (APOT) quantization is an extension of POT quantization that introduces an additive term in addition to the power-of-two scale. This allows for a more flexible representation of the quantized values, potentially reducing quantization error compared to pure POT.
# 
# The `apot_quantize_tensor` function provides a simplified implementation of this method. It calculates a scale and zero point based on the minimum and maximum values within each quantization group (or the entire tensor). The scale is derived from the range of values and the zero point is calculated to center the quantized values around zero. The weights are then quantized and dequantized using these calculated scales and zero points.
# 
# **Note:** The provided implementation is a simplified version for demonstration purposes. A full APOT implementation as described in the paper would typically involve a search process to find the optimal additive power-of-two scales that minimize quantization error.

# %%


# %%
del model
gc.collect()
torch.cuda.empty_cache()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
pot_quantize_model_weight(model, w_bit=3, q_group_size=128)

# Evaluate the model
model_perplexity = evaluate(model, tokenizer)
model_size = get_model_size(model, data_width=3, group_size=128)
print(f"\nmodel perplexity: {model_perplexity:.2f}")
print(f"model size: {model_size/MiB:.2f} MiB")

# %% [markdown]
# ## Summary:
# 
# ### Data Analysis Key Findings
# 
# *   **Model Size:** Implementing 3-bit quantization with a group size of 128 for both POT and APOT consistently resulted in a quantized model size of approximately 495.06 MiB, a significant reduction compared to the original FP32 model size (which would be around 1300 MiB for 1.3B parameters).
# *   **POT Perplexity:** The Power of Two (POT) quantization method, with 3-bit weights and a group size of 128, resulted in a very high model perplexity of 16368.73, indicating significant performance degradation.
# *   **APOT Perplexity:** The Additive Power of Two (APOT) quantization method, with the same 3-bit weights and 128 group size, yielded a much lower perplexity of 121.90 compared to pure POT, suggesting improved accuracy preservation.
# *   **Comparison:** APOT quantization achieved a significantly better perplexity than pure POT for the same bit width and group size, demonstrating the benefit of the additive term in reducing quantization error.
# 
# ### Insights or Next Steps
# 
# *   The large difference in perplexity between POT and APOT highlights the importance of flexible scaling and zero point adjustments in low-bit quantization to minimize accuracy loss. Pure power-of-two scaling alone may be too restrictive for achieving acceptable performance at very low bit widths.
# *   Further optimization of the APOT method, potentially involving a search for optimal additive power-of-two scales as described in the paper, could potentially lead to even better perplexity results while maintaining the quantized model size.
# 

# %% [markdown]
# # Claue suggested apot fix:

# %%
del model
gc.collect()
torch.cuda.empty_cache()

# %%

def apot_quantize_tensor(w, n_bit=4, q_group_size=-1, k=2):
    """
    Additive Power-of-Two quantization following the APoT paper (ICLR 2020).

    Key formulation (Equation 5 from paper):
    Q_a(α, kn) = γ × {Σ p_i} where p_i ∈ {0, 1/2^i, 1/2^(i+n), ..., 1/2^(i+(2^k-2)n)}

    Args:
        w: Weight tensor to quantize
        n_bit: Total number of bits for quantization
        q_group_size: Group size for quantization (-1 for per-channel)
        k: Base bit-width (default=2, as used in paper)
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate number of additive terms: n = floor(b/k)
    n = n_bit // k

    # Handle edge case for very low bits
    if n < 1:
        n = 1

    # Generate all possible quantization levels according to Equation 5
    # For unsigned values first, we'll handle signs separately
    levels = generate_apot_levels(n, k, device=w.device)

    # Compute scaling factor γ
    # γ is chosen so that max level equals 1.0 (we'll scale by alpha later)
    max_level = levels.max()
    gamma = 1.0 / max_level if max_level > 0 else 1.0
    levels = levels * gamma

    # Convert to full quantization levels (including negative values)
    # Full levels: {-levels, 0, +levels}
    positive_levels = levels[levels > 0]
    full_levels = torch.cat([
        -positive_levels.flip(0),
        torch.tensor([0.0], device=w.device),
        positive_levels
    ])

    # Step 1: Initialize base scale
    max_val = w.abs().amax(dim=1, keepdim=True)
    s_0 = max_val.clone()
    s_0 = torch.clamp(s_0, min=1e-5)

    # Step 2: Grid search for optimal scale multiplier
    B = torch.arange(0.01, 2.01, 0.01, device=w.device)
    best_error = torch.full((w.size(0), 1), float('inf'), device=w.device)
    best_scale = s_0.clone()

    for b in B:
        # Candidate scale
        s_b = s_0 * b

        # Quantize with this scale
        w_normalized = w / s_b

        # Find closest quantization level for each weight
        # Using broadcasting: w_normalized is [rows, cols], full_levels is [num_levels]
        distances = torch.abs(
            w_normalized.unsqueeze(-1) - full_levels.view(1, 1, -1)
        )
        closest_idx = torch.argmin(distances, dim=-1)
        w_q_normalized = full_levels[closest_idx]

        # Reconstruct
        w_q = s_b * w_q_normalized

        # Compute MSE
        error = ((w - w_q) ** 2).sum(dim=1, keepdim=True)

        # Update best scale
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, s_b, best_scale)

    # Final quantization with best scale
    with torch.no_grad():
        w_normalized = w / best_scale
        distances = torch.abs(
            w_normalized.unsqueeze(-1) - full_levels.view(1, 1, -1)
        )
        closest_idx = torch.argmin(distances, dim=-1)
        w_q_normalized = full_levels[closest_idx]
        w_quantized = best_scale * w_q_normalized

    assert torch.isnan(w_quantized).sum() == 0
    w_quantized = w_quantized.reshape(org_w_shape)

    return w_quantized


def generate_apot_levels(n, k, device='cpu'):
    """
    Generate APOT quantization levels according to Equation 5.

    Q_a(α, kn) = γ × {Σ p_i} where p_i ∈ {0, 1/2^i, 1/2^(i+n), ..., 1/2^(i+(2^k-2)n)}

    Args:
        n: Number of additive terms
        k: Base bit-width (bits per term)
        device: Device to create tensors on

    Returns:
        Tensor of all possible quantization levels (unsigned)
    """
    # Each p_i can take 2^k values
    num_choices_per_term = 2 ** k

    # Generate possible values for each additive term
    all_p_values = []
    for i in range(n):
        # p_i ∈ {0, 1/2^i, 1/2^(i+n), 1/2^(i+2n), ..., 1/2^(i+(2^k-2)n)}
        p_i_values = []
        for j in range(num_choices_per_term):
            if j == 0:
                p_i_values.append(0.0)
            else:
                exponent = i + (j - 1) * n
                p_i_values.append(2.0 ** (-exponent))
        all_p_values.append(p_i_values)

    # Generate all combinations (Cartesian product)
    # Total number of levels = (2^k)^n = 2^(kn) = 2^n_bit
    import itertools
    all_combinations = list(itertools.product(*all_p_values))

    # Sum up each combination to get final levels
    levels = torch.tensor([sum(combo) for combo in all_combinations], device=device)

    # Remove duplicates and sort
    levels = torch.unique(levels)
    levels = torch.sort(levels)[0]

    return levels


@torch.no_grad()
def apot_quantize_model_weight(model, w_bit, q_group_size, k=2):
    """
    Apply APOT quantization to all linear layers in the model.

    Args:
        model: The model to quantize
        w_bit: Number of bits for weight quantization
        q_group_size: Group size for quantization
        k: Base bit-width (default=2 as in paper)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data = apot_quantize_tensor(
                module.weight.data,
                n_bit=w_bit,
                q_group_size=q_group_size,
                k=k
            )
    return model

# Example usage and evaluation code
def evaluate_apot_example():
    """
    Example showing how to use APOT quantization with different configurations.
    Assumes you have evaluate() and get_model_size() functions defined.
    """
    # Configuration
    MiB = 1024 * 1024
    model_path = "facebook/opt-1.3b"

    # Example 1: 4-bit APOT with k=2 (2 additive terms)
    print("\n=== 4-bit APOT (k=2, n=2 addends) ===")
    # if(model):
    #   del model
    # gc.collect()
    # torch.cuda.empty_cache()

    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    # apot_quantize_model_weight(model, w_bit=4, q_group_size=128, k=2)
    # model_perplexity = evaluate(model, tokenizer)
    # model_size = get_model_size(model, data_width=4, group_size=128)
    # print(f"Perplexity: {model_perplexity:.2f}")
    # print(f"Model size: {model_size/MiB:.2f} MiB")

    # Example 2: 2-bit APOT with k=2 (1 addend)
    print("\n=== 2-bit APOT (k=2, n=1 addend) ===")
    # if(model):
    #   del model
    gc.collect()
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    apot_quantize_model_weight(model, w_bit=2, q_group_size=128, k=2)
    model_perplexity = evaluate(model, tokenizer)
    model_size = get_model_size(model, data_width=2, group_size=128)
    print(f"Perplexity: {model_perplexity:.2f}")
    print(f"Model size: {model_size/MiB:.2f} MiB")

    # Example 3: 3-bit APOT (special case for odd bits)
    print("\n=== 3-bit APOT (k=2, n=1 addend + 1 special term) ===")
    # For 3-bit, need to handle specially as per Equation 6 in paper


if __name__ == "__main__":
    # Test level generation
    print("Testing APOT level generation:")
    print("\n4-bit (k=2, n=2):")
    levels_4bit = generate_apot_levels(n=2, k=2)
    print(f"Number of levels: {len(levels_4bit)}")
    print(f"Levels: {levels_4bit[:10]}...")  # Show first 10

    print("\n2-bit (k=2, n=1):")
    levels_2bit = generate_apot_levels(n=1, k=2)
    print(f"Number of levels: {len(levels_2bit)}")
    print(f"Levels: {levels_2bit}")

    # evaluate_apot_example()

# %%


# %%
# =============================================================================
# POT QUANTIZATION (WEIGHT-ONLY)
# =============================================================================

def pot_quantize_tensor(w, n_bit=4, q_group_size=-1):
    """
    Power-of-Two quantization following the POT paper.

    Represents weights as: w_q = scale * sign(w) * 2^E
    Uses grid search for optimal scale initialization.

    Args:
        w: Weight tensor to quantize
        n_bit: Number of bits for quantization
        q_group_size: Group size for quantization (-1 for per-channel)
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Maximum exponent for n-bit quantization
    q_max = 2**(n_bit - 1) - 1

    # Step 1: Initialize base scale
    max_val = w.abs().amax(dim=1, keepdim=True)
    s_0 = max_val / (2**q_max)
    s_0 = torch.clamp(s_0, min=1e-5)

    # Step 2: Grid search for optimal scale multiplier
    B = torch.arange(0.01, 2.01, 0.01, device=w.device)
    best_error = torch.full((w.size(0), 1), float('inf'), device=w.device)
    best_scale = s_0.clone()

    for b in B:
        s_b = s_0 * b

        # Compute exponent: E = clamp(round(log2(|w| / s_b)), 0, q_max)
        with torch.no_grad():
            ratio = torch.clamp(w.abs() / s_b, min=1e-10)
            E = torch.clamp(torch.round(torch.log2(ratio)), 0, q_max)

        # Reconstruct: w_q = s_b * sign(w) * 2^E
        w_q = s_b * torch.sign(w) * torch.pow(2.0, E)

        # Compute MSE
        error = ((w - w_q) ** 2).sum(dim=1, keepdim=True)

        # Update best scale
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, s_b, best_scale)

    # Final quantization with best scale
    with torch.no_grad():
        ratio = torch.clamp(w.abs() / best_scale, min=1e-10)
        E = torch.clamp(torch.round(torch.log2(ratio)), 0, q_max)

    w_quantized = best_scale * torch.sign(w) * torch.pow(2.0, E)

    assert torch.isnan(w_quantized).sum() == 0
    w_quantized = w_quantized.reshape(org_w_shape)

    return w_quantized


# =============================================================================
# POT WITH ACTIVATION-AWARE SCALING
# =============================================================================

@torch.no_grad()
def pot_quantize_tensor_with_input(w, x, n_bit=4, q_group_size=-1):
    """
    POT quantization with activation-aware scale selection.
    Minimizes reconstruction error: ||Q(W) * X - W * X||

    Args:
        w: Weight tensor [out_features, in_features]
        x: Input activations [batch, seq_len, in_features]
        n_bit: Number of bits
        q_group_size: Group size for quantization
    """
    org_w_shape = w.shape

    # Reshape for group quantization
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Prepare input: flatten batch and sequence dimensions
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])

    # Maximum exponent
    q_max = 2**(n_bit - 1) - 1

    # Base scale initialization
    max_val = w.abs().amax(dim=1, keepdim=True)
    s_0 = max_val / (2**q_max)
    s_0 = torch.clamp(s_0, min=1e-5)

    # Compute original output: W @ X^T
    w_device = w.device
    x_sample = x[:min(128, x.shape[0])].to(w_device)  # Use subset for efficiency
    org_out = w @ x_sample.T

    # Grid search considering reconstruction error
    B = torch.arange(0.1, 2.0, 0.1, device=w.device)  # Coarser grid for speed
    best_error = torch.full((w.size(0),), float('inf'), device=w.device)
    best_scale = s_0.clone()

    for b in B:
        s_b = s_0 * b

        # Quantize with this scale
        ratio = torch.clamp(w.abs() / s_b, min=1e-10)
        E = torch.clamp(torch.round(torch.log2(ratio)), 0, q_max)
        w_q = s_b * torch.sign(w) * torch.pow(2.0, E)

        # Compute reconstruction error
        quant_out = w_q @ x_sample.T
        error = ((org_out - quant_out) ** 2).mean(dim=1)

        # Update best
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, s_b.squeeze(1), best_scale.squeeze(1)).unsqueeze(1)

    # Final quantization
    ratio = torch.clamp(w.abs() / best_scale, min=1e-10)
    E = torch.clamp(torch.round(torch.log2(ratio)), 0, q_max)
    w_quantized = best_scale * torch.sign(w) * torch.pow(2.0, E)

    assert torch.isnan(w_quantized).sum() == 0
    return w_quantized.reshape(org_w_shape)


@torch.no_grad()
def pot_quantize_model_weight(model, w_bit, q_group_size, input_feat=None):
    """
    Apply POT quantization to all linear layers.

    Args:
        model: Model to quantize
        w_bit: Number of bits
        q_group_size: Group size
        input_feat: Optional calibration data for activation-aware quantization
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if input_feat is not None and name in input_feat:
                # Use activation-aware quantization
                x = torch.cat([inp.unsqueeze(0) for inp in input_feat[name]], dim=0)
                if x.dim() == 4:  # Handle [batch, 1, seq, hidden]
                    x = x.squeeze(1)
                module.weight.data = pot_quantize_tensor_with_input(
                    module.weight.data, x, n_bit=w_bit, q_group_size=q_group_size
                )
            else:
                # Weight-only quantization
                module.weight.data = pot_quantize_tensor(
                    module.weight.data, n_bit=w_bit, q_group_size=q_group_size
                )
    return model


# =============================================================================
# APOT QUANTIZATION (FIXED VERSION)
# =============================================================================

def generate_apot_levels(n, k, device='cpu'):
    """
    Generate APOT quantization levels according to the paper.

    Each level is sum of n power-of-two terms, where each term has 2^k choices.

    Args:
        n: Number of additive terms
        k: Base bit-width (bits per term)
        device: Device to create tensors on

    Returns:
        Tensor of unique quantization levels (unsigned)
    """
    num_choices_per_term = 2 ** k

    # Generate possible values for each term
    all_p_values = []
    for i in range(n):
        # Each term can be: 0, or 2^(-i), 2^(-(i+n)), 2^(-(i+2n)), ...
        p_i_values = [0.0]  # Include zero
        for j in range(1, num_choices_per_term):
            exponent = i + (j - 1) * n
            p_i_values.append(2.0 ** (-exponent))
        all_p_values.append(p_i_values)

    # Generate all combinations
    all_combinations = list(itertools.product(*all_p_values))

    # Sum each combination
    levels = torch.tensor([sum(combo) for combo in all_combinations],
                          dtype=torch.float32, device=device)

    # Remove duplicates and sort
    levels = torch.unique(levels)
    levels = torch.sort(levels)[0]

    return levels


def apot_quantize_tensor(w, n_bit=4, q_group_size=-1, k=2):
    """
    Additive Power-of-Two quantization (fixed version).

    Represents weights as: w_q = scale * sum(sign_i * 2^E_i)

    Args:
        w: Weight tensor to quantize
        n_bit: Total number of bits
        q_group_size: Group size for quantization
        k: Base bit-width (default=2)
    """
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Calculate number of additive terms
    n = max(1, n_bit // k)

    # Generate quantization levels
    levels = generate_apot_levels(n, k, device=w.device)

    # Normalize levels
    max_level = levels.max()
    if max_level > 0:
        levels = levels / max_level

    # Create symmetric levels: {-levels, 0, +levels}
    positive_levels = levels[levels > 0]
    full_levels = torch.cat([
        -positive_levels.flip(0),
        torch.tensor([0.0], device=w.device),
        positive_levels
    ])

    # Initialize base scale
    max_val = w.abs().amax(dim=1, keepdim=True)
    s_0 = max_val.clone()
    s_0 = torch.clamp(s_0, min=1e-5)

    # Grid search for optimal scale
    B = torch.arange(0.01, 2.01, 0.05, device=w.device)  # Adjusted step
    best_error = torch.full((w.size(0), 1), float('inf'), device=w.device)
    best_scale = s_0.clone()

    for b in B:
        s_b = s_0 * b

        # Normalize weights
        w_normalized = w / s_b

        # Find closest level for each weight (vectorized)
        # w_normalized: [rows, cols], full_levels: [num_levels]
        distances = torch.abs(
            w_normalized.unsqueeze(-1) - full_levels.view(1, 1, -1)
        )
        closest_idx = torch.argmin(distances, dim=-1)
        w_q_normalized = full_levels[closest_idx]

        # Reconstruct
        w_q = s_b * w_q_normalized

        # Compute MSE
        error = ((w - w_q) ** 2).sum(dim=1, keepdim=True)

        # Update best
        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, s_b, best_scale)

    # Final quantization with best scale
    with torch.no_grad():
        w_normalized = w / best_scale
        distances = torch.abs(
            w_normalized.unsqueeze(-1) - full_levels.view(1, 1, -1)
        )
        closest_idx = torch.argmin(distances, dim=-1)
        w_q_normalized = full_levels[closest_idx]
        w_quantized = best_scale * w_q_normalized

    assert torch.isnan(w_quantized).sum() == 0
    w_quantized = w_quantized.reshape(org_w_shape)

    return w_quantized


@torch.no_grad()
def apot_quantize_tensor_with_input(w, x, n_bit=4, q_group_size=-1, k=2):
    """
    APOT quantization with activation-aware scale selection.
    """
    org_w_shape = w.shape

    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)

    assert w.dim() == 2

    # Generate APOT levels
    n = max(1, n_bit // k)
    levels = generate_apot_levels(n, k, device=w.device)
    max_level = levels.max()
    if max_level > 0:
        levels = levels / max_level

    positive_levels = levels[levels > 0]
    full_levels = torch.cat([
        -positive_levels.flip(0),
        torch.tensor([0.0], device=w.device),
        positive_levels
    ])

    # Prepare input
    if x.dim() == 3:
        x = x.reshape(-1, x.shape[-1])

    w_device = w.device
    x_sample = x[:min(128, x.shape[0])].to(w_device)
    org_out = w @ x_sample.T

    # Scale search with reconstruction error
    max_val = w.abs().amax(dim=1, keepdim=True)
    s_0 = torch.clamp(max_val, min=1e-5)

    B = torch.arange(0.1, 2.0, 0.1, device=w.device)
    best_error = torch.full((w.size(0),), float('inf'), device=w.device)
    best_scale = s_0.clone()

    for b in B:
        s_b = s_0 * b

        # Quantize
        w_normalized = w / s_b
        distances = torch.abs(w_normalized.unsqueeze(-1) - full_levels.view(1, 1, -1))
        closest_idx = torch.argmin(distances, dim=-1)
        w_q = s_b * full_levels[closest_idx]

        # Reconstruction error
        quant_out = w_q @ x_sample.T
        error = ((org_out - quant_out) ** 2).mean(dim=1)

        mask = error < best_error
        best_error = torch.where(mask, error, best_error)
        best_scale = torch.where(mask, s_b.squeeze(1), best_scale.squeeze(1)).unsqueeze(1)

    # Final quantization
    w_normalized = w / best_scale
    distances = torch.abs(w_normalized.unsqueeze(-1) - full_levels.view(1, 1, -1))
    closest_idx = torch.argmin(distances, dim=-1)
    w_quantized = best_scale * full_levels[closest_idx]

    assert torch.isnan(w_quantized).sum() == 0
    return w_quantized.reshape(org_w_shape)


@torch.no_grad()
def apot_quantize_model_weight(model, w_bit, q_group_size, k=2, input_feat=None):
    """
    Apply APOT quantization to all linear layers.

    Args:
        model: Model to quantize
        w_bit: Number of bits
        q_group_size: Group size
        k: Base bit-width (default=2)
        input_feat: Optional calibration data
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if input_feat is not None and name in input_feat:
                # Activation-aware quantization
                x = torch.cat([inp.unsqueeze(0) for inp in input_feat[name]], dim=0)
                if x.dim() == 4:
                    x = x.squeeze(1)
                module.weight.data = apot_quantize_tensor_with_input(
                    module.weight.data, x, n_bit=w_bit,
                    q_group_size=q_group_size, k=k
                )
            else:
                # Weight-only quantization
                module.weight.data = apot_quantize_tensor(
                    module.weight.data, n_bit=w_bit,
                    q_group_size=q_group_size, k=k
                )
    return model


# =============================================================================
# EVALUATION CODE
# =============================================================================

def evaluate_pot_apot_quantization(model_path, tokenizer, evaluate_fn, get_model_size_fn):
    """
    Complete evaluation pipeline for POT and APOT quantization.

    Args:
        model_path: Path to pretrained model
        tokenizer: Model tokenizer
        evaluate_fn: Function to compute perplexity
        get_model_size_fn: Function to compute model size
    """
    MiB = 1024 * 1024

    # =========================================================================
    # Step 1: Collect calibration data
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Collecting Calibration Data")
    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", offload_folder="offload")
    input_feat = get_calib_feat(model, tokenizer)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Step 2: POT Quantization (Weight-only)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: POT Quantization (Weight-only, 3-bit)")
    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", offload_folder="offload")
    pot_quantize_model_weight(model, w_bit=3, q_group_size=128, input_feat=None)

    model_perplexity = evaluate_fn(model, tokenizer)
    model_size = get_model_size_fn(model, data_width=3, group_size=128)
    print(f"\nPOT (weight-only) Results:")
    print(f"  Perplexity: {model_perplexity:.2f}")
    print(f"  Model size: {model_size/MiB:.2f} MiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Step 3: POT Quantization (Activation-aware)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: POT Quantization (Activation-aware, 3-bit)")
    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", offload_folder="offload")
    pot_quantize_model_weight(model, w_bit=3, q_group_size=128, input_feat=input_feat)

    model_perplexity = evaluate_fn(model, tokenizer)
    model_size = get_model_size_fn(model, data_width=3, group_size=128)
    print(f"\nPOT (activation-aware) Results:")
    print(f"  Perplexity: {model_perplexity:.2f}")
    print(f"  Model size: {model_size/MiB:.2f} MiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Step 4: APOT Quantization (Weight-only)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: APOT Quantization (Weight-only, 4-bit, k=2)")
    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", offload_folder="offload")
    apot_quantize_model_weight(model, w_bit=4, q_group_size=128, k=2, input_feat=None)

    model_perplexity = evaluate_fn(model, tokenizer)
    model_size = get_model_size_fn(model, data_width=4, group_size=128)
    print(f"\nAPOT (weight-only) Results:")
    print(f"  Perplexity: {model_perplexity:.2f}")
    print(f"  Model size: {model_size/MiB:.2f} MiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # Step 5: APOT Quantization (Activation-aware)
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: APOT Quantization (Activation-aware, 4-bit, k=2)")
    print("="*80)

    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", offload_folder="offload")
    apot_quantize_model_weight(model, w_bit=4, q_group_size=128, k=2, input_feat=input_feat)

    model_perplexity = evaluate_fn(model, tokenizer)
    model_size = get_model_size_fn(model, data_width=4, group_size=128)
    print(f"\nAPOT (activation-aware) Results:")
    print(f"  Perplexity: {model_perplexity:.2f}")
    print(f"  Model size: {model_size/MiB:.2f} MiB")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("Evaluation Complete!")
    print("="*80)

model_path = "facebook/opt-1.3b"

evaluate_pot_apot_quantization(
    model_path=model_path,
    tokenizer=tokenizer,
    evaluate_fn=evaluate,  # Your existing evaluate function
    get_model_size_fn=get_model_size  # Your existing get_model_size function
)

## NOTE: THE APOT IMPLEMENTATION HERE HAS FLAWS- ensure correct APOT implementation is used