from torch.utils.checkpoint import checkpoint

def enable_activation_checkpointing(model):
    wrapped=[]
    for prefix,layers in [('ms_encoder.transformer_encoder.layers',model.ms_encoder.transformer_encoder.layers),('graph_encoder.tf_layers',model.graph_encoder.tf_layers)]:
        for i,layer in enumerate(layers):
            cls=type(layer)
            if not getattr(cls,'_encoder_checkpoint_wrapper',False):
                original=cls.forward
                def forward(self,*args,_original=original,**kwargs):
                    if self.training and getattr(self,'_encoder_checkpoint_enabled',False):
                        def run(*values):return _original(self,*values,**kwargs)
                        return checkpoint(run,*args,use_reentrant=False,preserve_rng_state=True)
                    return _original(self,*args,**kwargs)
                cls.forward=forward
                cls._encoder_checkpoint_wrapper=True
            layer._encoder_checkpoint_enabled=True
            wrapped.append(prefix+'.'+str(i))
    return wrapped
