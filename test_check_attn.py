from src.train.teacher_attention import TeacherAttentionCache

p = TeacherAttentionCache("dataset/attn_cache_mmdocir_phi3_prior_full")
q = TeacherAttentionCache("dataset/attn_cache_mmdocir_phi3_query_full")
ps = set(p.vectors)
qs = set(q.vectors)
print("prior:", len(ps))
print("query:", len(qs))
print("overlap:", len(ps & qs))
print("query missing vs prior:", len(ps - qs))
print("query extra vs prior:", len(qs - ps))
