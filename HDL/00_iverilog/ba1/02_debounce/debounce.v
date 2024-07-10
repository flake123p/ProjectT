/*
Module:
	Switch Level
	Gate Level
	Data Level
	Behavioral Level(Algorithmic)
	
	Data Level + Behavioral Level => RTL (Register transfer level)
*/

// Port List = Parameters. Input at beginning, output at ending.
module debounce(clk, rstn, but_in, but_deb_o);
input           clk;
input           rstn;
input           but_in;
output          but_deb_o;

parameter s0 = 3'h0, s1 = 3'h1, s2 = 3'h2,
          s3 = 3'h3, s4 = 3'h4, s5 = 3'h5;

reg[2:0] cs;// current state
reg[2:0] ns;// next state
reg[31:0] cnt;

always @(posedge clk or negedge rstn) begin
    if (~rstn)
        cnt <= 3'b000;
    else if (cnt == 32'd999999)
        cnt <= 32'h0;
    else if (cs == s1)
        cnt <= cnt + 32'h1;
    else if (cs == s4)
        cnt <= cnt + 32'h1;
    else
        cnt <= cnt;
end

always @(posedge clk or negedge rstn) begin
    if (~rstn)
        cs <= s0;
    else
        cs <= ns;
end

always @(*) begin
    case (cs)
        s0: 
            if (but_in == 1'b0)
                ns = s1;
            else
                ns = s0;
        s1:
            if (cnt == 32'd999999)
                ns = s2;
            else
                ns = s1;
        s2:
            if (but_in == 1'b1)
                ns = s0; //button can't mantain low ...
            else
                ns = s3;
        s3: if (but_in == 1'b1)
                ns = s4;
            else
                ns = s3;
        s4:
            if (cnt == 32'd999999)
                ns = s5;
            else
                ns = s4;
        s5:
            if (but_in == 1'b1)
                ns = s0;
            else
                ns = s3;
        default:
            ns = s0;
    endcase
end

assign but_deb_o = 
    (cs == s0) ? 1'b1 :
    (cs == s1) ? 1'b1 :
    (cs == s2) ? 1'b1 :
    (cs == s3) ? 1'b0 :
    (cs == s4) ? 1'b0 :
    (cs == s5) ? 1'b0 :
    1'b0;

endmodule