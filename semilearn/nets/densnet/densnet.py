import torch.nn as nn
import torch
import timm

class DenseNet121(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DenseNet121, self).__init__()
        self.model = timm.create_model('densenet121', pretrained=True,num_classes=num_classes)
        self.num_features = self.model.classifier.in_features
    def forward(self, x,only_fc=False,only_feat=False):
        if only_fc:
            logits = self.model.classifier(x)
            return logits

        x = self.model.forward_features(x)
        x = self.model.global_pool(x)
        x = torch.flatten(x, 1)

        if only_feat:
            return x

        out = self.model.classifier(x)
        result_dict = {'logits':out, 'feat':x}
        return result_dict

        # feat = self.model.forward_features(x)
        # pre_logits = self.model.global_pool(feat)
        # pre_logits = torch.flatten(pre_logits, 1)
        # logits = self.model.classifier(pre_logits)
        # return {'logits': logits, 'feat': feat}

def densenet121(num_classes, pretrained=True):
    return DenseNet121(num_classes, pretrained)
if __name__ == '__main__':
    model = DenseNet121(10,False)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(output['logits'].shape)
    print(output['feat'].shape)
