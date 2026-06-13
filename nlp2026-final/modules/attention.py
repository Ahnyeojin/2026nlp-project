import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # key, value, query??????좏삎蹂??layer 珥덇린??
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # ???쒕∼?꾩썐? ?몃옖?ㅽ룷癒몄쓽 ?먮옒 援ы쁽???곕씪 normalized attention scores???곸슜?쒕떎.
    # ?ㅼ냼 ?대??곸씠吏留? 寃쏀뿕?곸쑝濡??닿쾬?????섏? ?깅뒫???쒓났?쒕떎怨??뚮젮???덈떎.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # hidden_state (x) 瑜??ъ쁺?섍린 ?꾪빐 k, v, q???대떦 linear_layer媛 ?ъ슜?쒕떎.
    proj = linear_layer(x)
    # ?ㅼ쓬?쇰줈, ?꾨줈?앹뀡??????щ윭 ?ㅻ뱶瑜??앹꽦?댁빞 ?쒕떎. 
    # ?대뒗 ????곹깭瑜?self.num_attention_heads濡?遺꾪븷?섎ŉ, 
    # 媛??ㅻ뱶??self.attention_head_size ?ш린瑜?媛뽯룄濡??쒕떎.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # ?곸젅???꾩튂?섏뿬 ?ш린 [bs, num_attention_heads, seq_len, attention_head_size]???꾨줈?앹뀡???삳뒗??
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### ?꾩꽦?쒖폒????鍮?肄붾뱶 釉붾줉
    # raise NotImplementedError
  
    # ref: Lecture Note (09.Transformers-2)
      
    # 1. Attention Score (Lecture Note 29p)
    # calculate QK^T)
    attention_scores = torch.matmul(query, key.transpose(-1, -2)) 

    # 2. Scaled dot product (Lecture Note 30p)
    # divide by sqrt{d_k}
    attention_scores = attention_scores / (self.attention_head_size ** 0.5)

    # 3. Causal mask & Padding mask
    seq_len = attention_scores.size(-1)

    # Causal mask (future token mask)
    # create causal mask (line 55 - Gemini used)
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=attention_scores.device), diagonal=1).bool()
    attention_scores = attention_scores.masked_fill(causal_mask, -1e4)  # FP16 AMP compatible

    # Padding mask
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    # 4. Softmax
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    attention_probs = self.dropout(attention_probs)

    # 5. Attention value
    attention_val = torch.matmul(attention_probs, value)
      
    # 6. Merge (reverse of transform function)        
    attention_val = rearrange(attention_val, 'b h t d -> b t h d')
    attention_val = rearrange(attention_val, 'b t h d -> b t (h d)')

    return attention_val


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # 癒쇱?, self.transform???ъ슜?섏뿬 multi-head attention???꾩슂??
    # 媛??좏겙??key, value, query瑜??앹꽦?댁빞 ?쒕떎(?⑥닔 ?대????먯꽭???댁슜 ?덉쓬).
    # *_layer???ш린 = [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # multi-head attention 怨꾩궛.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
