{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMFYBVDM3MJV9KCAA64COjr",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/hiteshJindal/Python_Projects/blob/master/allconv.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hWm851s16wXz"
      },
      "outputs": [],
      "source": [
        "import math\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "\n",
        "\n",
        "class GELU(nn.Module):\n",
        "    def __init__(self):\n",
        "        super(GELU, self).__init__()\n",
        "\n",
        "    def forward(self, x):\n",
        "        return F.relu(x, inplace=True)\n",
        "        # return torch.sigmoid(1.702 * x) * x\n",
        "        # return 0.5 * x * (1 + torch.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))\n",
        "\n",
        "\n",
        "def make_layers(cfg):\n",
        "    layers = []\n",
        "    in_channels = 3\n",
        "    for v in cfg:\n",
        "        if v == 'Md':\n",
        "            layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.5)]\n",
        "        elif v == 'A':\n",
        "            layers += [nn.AvgPool2d(kernel_size=8)]\n",
        "        elif v == 'NIN':\n",
        "            conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=1)\n",
        "            layers += [conv2d, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]\n",
        "        elif v == 'nopad':\n",
        "            conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)\n",
        "            layers += [conv2d, nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True)]\n",
        "        else:\n",
        "            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)\n",
        "            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]\n",
        "            in_channels = v\n",
        "    return nn.Sequential(*layers)\n",
        "\n",
        "\n",
        "class AllConvNet(nn.Module):\n",
        "    def __init__(self, num_classes):\n",
        "        super(AllConvNet, self).__init__()\n",
        "\n",
        "        self.num_classes = num_classes\n",
        "\n",
        "        # if num_classes > 10:\n",
        "        #     self.width1, w1 = 128, 128\n",
        "        #     self.width2, w2 = 256, 256\n",
        "        # else:\n",
        "        #     self.width1, w1 = 96,  96\n",
        "        #     self.width2, w2 = 192, 192\n",
        "\n",
        "        self.width1, w1 = 96,  96\n",
        "        self.width2, w2 = 192, 192\n",
        "\n",
        "        self.features = make_layers([w1, w1, w1, 'Md', w2, w2, w2, 'Md', 'nopad', 'NIN', 'NIN', 'A'])\n",
        "        self.classifier = nn.Linear(self.width2, num_classes)\n",
        "\n",
        "        for m in self.modules():\n",
        "            if isinstance(m, nn.Conv2d):\n",
        "                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels\n",
        "                m.weight.data.normal_(0, math.sqrt(2. / n))     # He initialization\n",
        "            elif isinstance(m, nn.BatchNorm2d):\n",
        "                m.weight.data.fill_(1)\n",
        "                m.bias.data.zero_()\n",
        "            elif isinstance(m, nn.Linear):\n",
        "                m.bias.data.zero_()\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.features(x)\n",
        "        x = F.avg_pool2d(x, 2)\n",
        "        x = x.view(x.size(0), -1)\n",
        "        x = self.classifier(x)\n",
        "        return x\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# New Section"
      ],
      "metadata": {
        "id": "0XDlZtVM7d_G"
      }
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "Czur_uWi69oS"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}