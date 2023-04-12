struct Input {
    [[vk::location(0)]] float3 Color : COLOR0;
    [[vk::location(1)]] float2 TexCoord : TEXCOORD0;
};

Texture2D textureColor : register(t1);
SamplerState samplerColor : register(s1);

struct Output {
    [[vk::location(0)]] float4 Color : COLOR0;
};

Output main(Input input) {
    Output output = (Output)0;

    output.Color = textureColor.Sample(samplerColor, input.TexCoord);

    return output;
}