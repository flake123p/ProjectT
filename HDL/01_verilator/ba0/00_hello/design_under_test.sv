module design_under_test(
    input clk, input rst,
    output logic valid,
    output logic [7:0] data
);
    logic [39:0] out_all;
    always_comb begin
        data = out_all[7:0];
        valid = (data != '0);
    end

    always_ff @(posedge clk or negedge rst) begin
        if (!rst) begin
            out_all <= "olleH";
        end else if (valid) begin
            out_all <= out_all >> 8;
        end
    end
endmodule