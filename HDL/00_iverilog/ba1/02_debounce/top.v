/*
Module:
	Switch Level
	Gate Level
	Data Level
	Behavioral Level(Algorithmic)
	
	Data Level + Behavioral Level => RTL (Register transfer level)
*/

// Port List = Parameters. Input at beginning, output at ending.
module top(clk, rstn, but_in, led_o);
input           clk;
input           rstn;
input           but_in;
output [3:0]    led_o;
wire            but_deb_o;
wire            but_deb_nedge;
reg             but_deb_o_1d;
reg[1:0] cs;// current state
reg[1:0] ns;// next state

debounce debounce (
.clk        (clk      ),
.rstn       (rstn     ),
.but_in     (but_in   ),
.but_deb_o  (but_deb_o)
);
always @(posedge clk or negedge rstn) begin
    if (~rstn)
        but_deb_o_1d <= 1'b0;
    else
        but_deb_o_1d <= but_deb_o;
end
assign but_deb_nedge = (but_deb_o) && (~but_deb_o_1d);

parameter s1 = 2'h0, s2 = 2'h1, s3 = 2'h2, s4 = 2'h3;

always @(posedge clk or negedge rstn) begin
    if (~rstn)
        cs <= s1;
    else
        cs <= ns;
end

always @(*) begin
    case (cs)
        s1:
            if (but_deb_nedge)
                ns = s2;
            else
                ns = s1;
        s2:
            if (but_deb_nedge)
                ns = s3;
            else
                ns = s2;
        s3: if (but_deb_nedge)
                ns = s4;
            else
                ns = s3;
        s4:
            if (but_deb_nedge)
                ns = s1;
            else
                ns = s4;
        default:
            ns = s1;
    endcase
end

assign led_o = 
    (cs == s1) ? 4'b0001 :
    (cs == s2) ? 4'b0010 :
    (cs == s3) ? 4'b0100 :
    (cs == s4) ? 4'b1000 :
    4'b0000;

endmodule