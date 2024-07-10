/*
Module:
	Switch Level
	Gate Level
	Data Level
	Behavioral Level(Algorithmic)
	
	Data Level + Behavioral Level => RTL (Register transfer level)
*/

// Port List = Parameters. Input at beginning, output at ending.
module counter(clk, rstn, cntout);
input          clk;
input          rstn;
output [2:0]   cntout;
reg    [2:0]   cntout;
wire   [2:0]   cnt_incr;

assign cnt_incr = cntout + 3'b001;

always @(posedge clk or negedge rstn) begin
   if (rstn == 1'b0)
      cntout <= 3'b000;
   else
      cntout <= cnt_incr;
end

endmodule