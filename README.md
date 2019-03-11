# pytorch-rnn-test
unit test for mkldnn RNN APIs on PyTorch.

usage
```python

# Add runtime switch for MKLDNN
# modify aten/src/ATen/native/TensorProperties.cpp
 bool mkldnn_is_acceptable(const Tensor& self) {
+  //user defined flag
+  int mkldnn_enabled = 1;
+  if (const char *env_p = std::getenv("MKLDNN_ENABLED")) {
+    mkldnn_enabled = std::stoi(env_p);
+  }
+  if (!mkldnn_enabled) return false;
+

# build from source
cd pytorch/src/dir
python setup.py install

# run test
python test_rnn.py
```
