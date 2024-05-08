# PinT-AmericanOptions
In the spirit of reproducible research, we provide here the MATLAB codes for replicate the results in our preprint:
"**Parallel-in-Time Iterative Methods for Pricing American Options**" by Xian-Ming Gu, Jun Liu, Cornelis W. Oosterlee, and Hui Xiao.

# Below is the short description of each code:
  -**AmericanOption_1D_BS.m**, driver code for Example 1 (1D Black-Scholes Model)  
  -**AmericanOption_2D_Spread.m**, driver code for Example 2 (2D Spread options)  
  -**AmericanOption_2D_Heston.m**, driver code for Example 3 (2D Heston Model)  
  -**LCP_policy.m**, the direct Policy iteration solver  
  -**LCP_policy_PinT.m**, the PinT Policy iteration solver using NKPA preconditioner  
  -**LCP_policy_block.m**, the Policy iteration solver based on reduced systems  
  -**LCP_policy_block_PinT.m**, the PinT Policy iteration solver based on reduced systems  
  -**LCP_policy_block_PinT_GMG.m**, the PinT Policy iteration solver based on reduced systems and geometric multigrid for step-(b)
  -**fgmres.m**, the flexible GMRES solver imported from the  TT Toolbox  
  -**mg_iter_2d.m**, the main geometric multigrid V-cycle iteration code  
  
  

Licence Information:

This library is free software. It can be redistributed and/or modified under the terms of the MIT License.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Copyright (c) 2024 by Jun Liu
