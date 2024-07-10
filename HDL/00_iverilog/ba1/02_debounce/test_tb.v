`timescale 1ns/1ns
module tb_counter;
reg clk;
reg rstn;
wire [2:0] cnt_val_o;

always begin
   #10 clk = ~clk;
end

initial begin
   $dumpfile("simple.vcd");
   $dumpvars(0, counter0);
   clk = 1'b0;
   rstn = 1'b0;
   #100 rstn = 1'b1;
   #500 $finish;
end

counter counter0(
   .clk   (clk      ),  
   .rstn  (rstn     ),
   .cntout(cnt_val_o)
);

endmodule