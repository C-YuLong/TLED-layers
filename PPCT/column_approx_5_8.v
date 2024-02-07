// LENGTH: 8
// THETA: 5

module column_approx_5_8(
	input clk,
	input [7:0] x,
	input [7:0] y,
	output [15:0] z
);

// the approximation
	wire [15:0] z_0;
	assign z_0 = {8'b0,((x>>5)<<5)&{8{y[0]}}};
	wire [15:0] z_1;
	assign z_1 = {8'b0,((x>>4)<<4)&{8{y[1]}}};
	wire [15:0] z_2;
	assign z_2 = {8'b0,((x>>3)<<3)&{8{y[2]}}};
	wire [15:0] z_3;
	assign z_3 = {8'b0,((x>>2)<<2)&{8{y[3]}}};
	wire [15:0] z_4;
	assign z_4 = {8'b0,((x>>1)<<1)&{8{y[4]}}};

	wire [15:0] z_5;
	assign z_5 = {8'b0,x&{8{y[5]}}};
	wire [15:0] z_6;
	assign z_6 = {8'b0,x&{8{y[6]}}};
	wire [15:0] z_7;
	assign z_7 = {8'b0,x&{8{y[7]}}};

  assign z = z_0+(z_1<<1)+(z_2<<2)+(z_3<<3)+(z_4<<4)+(z_5<<5)+(z_6<<6)+(z_7<<7);
endmodule
