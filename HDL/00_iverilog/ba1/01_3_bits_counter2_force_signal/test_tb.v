`timescale 1ns/1ns
module tb_counter;
reg clk;
reg rstn;
wire [2:0] cnt_val_o;

always begin
   #10 clk = ~clk;
end

counter counter0(
   .clk   (clk      ),  
   .rstn  (rstn     ),
   .cntout(cnt_val_o)
);

//
// Use wire & force to probe internal signal
//
wire [31:0] pw;
wire [2:0] pw2;

//
// Use reg & logic to do more jobs !!
//
reg [31:0] probe;
always @(posedge clk or negedge rstn) begin
   if (rstn == 1'b0) 
	   probe <= 'b0;
   else
      probe <= pw;
end

initial begin
   $dumpfile("simple.vcd");
   // $dumpvars(0, counter0);
   $dumpvars(0, tb_counter);
   clk = 1'b0;
   rstn = 1'b0;
   force     pw = counter0.internal_reg;
   force     pw2 = counter0.cnt_incr;
   #100 rstn = 1'b1;
   #500 $finish;
end

endmodule