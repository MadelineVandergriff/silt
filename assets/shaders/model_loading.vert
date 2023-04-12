struct Input {
    [[vk::location(0)]] float3 Position : POSITION0;
    [[vk::location(1)]] float3 Color : COLOR0;
    [[vk::location(2)]] float2 TexCoord : TEXCOORD0;
};

struct UBO {
    float4x4 Model;
    float4x4 View;
    float4x4 Projection;
};

cbuffer ubo: register(b0, space0) { UBO ubo; }

struct Output {
    float4 Position: SV_POSITION;
    [[vk::location(0)]] float3 Color : COLOR0;
    [[vk::location(1)]] float2 TexCoord : TEXCOORD0;
}

Output main(Input input) {
    Output output = (Output)0;

    output.Position = mul(ubo.Projection, mul(ubo.View, mul(ubo.Model, float4(input.Position.xyz, 1.0))));
    output.Color = input.Color;
    output.TexCoord = input.TexCoord;

    return output;
}